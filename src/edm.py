"""
EDM (Elucidating the Design Space of Diffusion Models, Karras et al. 2022)
preconditioning, training, and sampling for score-based diffusion.
"""

import copy
import torch
import torch.nn as nn
import numpy as np


# ---------------------------------------------------------------------------
# EDM Preconditioning Wrapper
# ---------------------------------------------------------------------------

class EDMPrecond(nn.Module):
    """
    Wraps a raw denoiser network F_theta with EDM preconditioning.

    D_theta(x; sigma) = c_skip(sigma)*x + c_out(sigma)*F_theta(c_in(sigma)*x, c_noise(sigma))

    The raw network receives c_noise(sigma) = ln(sigma)/4 as its time/conditioning input.
    """

    def __init__(self, net, sigma_data=1.0):
        super().__init__()
        self.net = net
        self.sigma_data = sigma_data

    def c_skip(self, sigma):
        return self.sigma_data ** 2 / (sigma ** 2 + self.sigma_data ** 2)

    def c_out(self, sigma):
        return sigma * self.sigma_data / torch.sqrt(sigma ** 2 + self.sigma_data ** 2)

    def c_in(self, sigma):
        return 1.0 / torch.sqrt(sigma ** 2 + self.sigma_data ** 2)

    def c_noise(self, sigma):
        return torch.log(sigma) / 4.0

    def forward(self, x_noisy, sigma):
        """
        x_noisy: (B, C, H, W)
        sigma:   (B,) noise levels
        """
        s = sigma
        if s.dim() == 0:
            s = s.unsqueeze(0).expand(x_noisy.shape[0])

        c_skip = self.c_skip(s)[:, None, None, None]
        c_out  = self.c_out(s)[:, None, None, None]
        c_in   = self.c_in(s)[:, None, None, None]
        c_noise = self.c_noise(s)  # (B,)

        F_out = self.net(c_in * x_noisy, c_noise)
        return c_skip * x_noisy + c_out * F_out


# ---------------------------------------------------------------------------
# EDM Loss Weight
# ---------------------------------------------------------------------------

def edm_loss_weight(sigma, sigma_data):
    """Per-sample loss weight: (sigma^2 + sigma_data^2) / (sigma * sigma_data)^2"""
    return (sigma ** 2 + sigma_data ** 2) / (sigma * sigma_data) ** 2


# ---------------------------------------------------------------------------
# Score Wrapper with Tikhonov Regularization
# ---------------------------------------------------------------------------

class EDMScoreWrapper(nn.Module):
    """
    Converts an EDM denoiser to a score function for use with SDEsampler.

    score = (D_theta(x; sigma) - x) / (sigma^2 + c_tikhonov)

    The SDEsampler passes flat (B, D) tensors and scalar-like t values.
    This wrapper converts t -> sigma via marginal_prob_std, reshapes to images,
    runs the denoiser, and reshapes back.
    """

    def __init__(self, edm_precond, marginal_prob_std, grid_size, c_tikhonov=0.0):
        super().__init__()
        self.edm_precond = edm_precond
        self.marginal_prob_std = marginal_prob_std
        self.grid_size = grid_size
        self.c_tikhonov = c_tikhonov

    def forward(self, x_flat, t):
        B, D = x_flat.shape
        N = self.grid_size
        x_img = x_flat.reshape(B, 1, N, N)

        sigma = self.marginal_prob_std(t)  # (B,)
        D_theta = self.edm_precond(x_img, sigma)  # (B, 1, N, N)

        denom = (sigma ** 2 + self.c_tikhonov)[:, None, None, None]
        score = (D_theta - x_img) / denom

        return score.reshape(B, D)


# ---------------------------------------------------------------------------
# Karras Sigma Schedule (for sampling)
# ---------------------------------------------------------------------------

def edm_sigma_schedule(n_steps, sigma_min=0.002, sigma_max=80.0, rho=7.0):
    """
    Karras et al. sigma schedule for sampling.
    Returns (n_steps+1,) tensor from sigma_max down to 0.
    """
    step_indices = torch.arange(n_steps + 1, dtype=torch.float64)
    inv_rho = 1.0 / rho
    sigmas = (sigma_max ** inv_rho + step_indices / n_steps *
              (sigma_min ** inv_rho - sigma_max ** inv_rho)) ** rho
    sigmas[-1] = 0  # final step is clean
    return sigmas.float()


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_edm(train_data_flat, grid_size, total_steps, checkpoint_at,
              base_channels=16, emb_dim=64, sigma_data=None,
              lr=1e-3, batch_size=8, P_mean=-1.2, P_std=1.2,
              c_tikhonov=0.0,
              seed=0, device="cpu", UNetClass=None):
    """
    Train a UNet with EDM preconditioning via denoising score matching.

    Loss: weight(sigma) * [ || D_theta(x_noisy; sigma) - x_0 ||^2
          + c_tikhonov * || (D_theta - x_noisy) ||^2 / sigma^2 ]

    The second term is a Tikhonov penalty designed so the stationary point
    matches the GMM closed-form Tikhonov score s* = s_true / (1 + c/sigma^2),
    i.e. the denominator changes from sigma^2 to sigma^2 + c.
    Ref: "Memorization and Regularization in GDM" (Baptista et al. 2025).

    Args:
        train_data_flat: (n_train, N*N) flattened training images
        grid_size: spatial size N (images are N x N)
        total_steps: number of training steps
        checkpoint_at: list of steps at which to save snapshots
        UNetClass: the raw UNet class (e.g. SmallUNet). Must accept
                   (base_channels, emb_dim) and forward(x, t).
        sigma_data: data std. If None, estimated from training data.
        c_tikhonov: Tikhonov constant (0.0 = standard EDM, >0 = regularized).

    Returns:
        dict {step -> EDMPrecond (eval mode, on device)}
    """
    torch.manual_seed(seed)

    if sigma_data is None:
        sigma_data = train_data_flat.std().item()
        print(f"Estimated sigma_data = {sigma_data:.4f}")

    unet = UNetClass(base_channels=base_channels, emb_dim=emb_dim).to(device)
    precond = EDMPrecond(unet, sigma_data=sigma_data).to(device)
    opt = torch.optim.Adam(precond.parameters(), lr=lr)

    n_train = train_data_flat.shape[0]
    N = grid_size
    saved = {}

    precond.train()
    for step in range(1, total_steps + 1):
        idx = torch.randint(0, n_train, (batch_size,), device=device)
        x0 = train_data_flat[idx].reshape(batch_size, 1, N, N)  # (B, 1, N, N)

        # Sample sigma from log-normal
        ln_sigma = torch.randn(batch_size, device=device) * P_std + P_mean
        sigma = ln_sigma.exp()  # (B,)

        # Add noise (VE-style: no mean scaling)
        noise = torch.randn_like(x0)
        x_noisy = x0 + sigma[:, None, None, None] * noise

        # Forward pass
        D_theta = precond(x_noisy, sigma)  # (B, 1, N, N)

        # Weighted denoising loss: weight(sigma) * ||D_theta - x_0||^2
        weight = edm_loss_weight(sigma, sigma_data)  # (B,)
        loss_per_sample = ((D_theta - x0) ** 2).mean(dim=(1, 2, 3))  # (B,)
        loss = (weight * loss_per_sample).mean()

        # Tikhonov penalty matching GMM Tikhonov (Baptista et al. 2025):
        # The GMM closed-form Tikhonov replaces the score denominator σ² with σ²+c,
        # giving s* = s_true / (1 + c/σ²).  The denoising term is λ(σ)||D−x₀||²,
        # so the penalty must also carry λ(σ) for it to cancel in the stationary
        # condition:  λ(D−x₀) + c·λ(D−x)/σ² = 0  →  s* = s_true/(1+c/σ²).
        if c_tikhonov > 0.0:
            s2 = sigma[:, None, None, None] ** 2
            score_penalty = ((D_theta - x_noisy) ** 2 / s2).mean(dim=(1, 2, 3))
            loss = loss + c_tikhonov * (weight * score_penalty).mean()

        opt.zero_grad()
        loss.backward()
        opt.step()

        if step % 500 == 0 or step == 1:
            print(f"  step {step}/{total_steps}  loss={loss.item():.6f}")

        if step in checkpoint_at:
            snap = copy.deepcopy(precond)
            snap.eval()
            saved[step] = snap.to(device)
            print(f"  >> checkpoint saved at step {step}")

    return saved
