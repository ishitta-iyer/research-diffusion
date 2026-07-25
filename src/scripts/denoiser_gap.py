"""Train/held-out denoiser gap test on the saved EDM UNet checkpoints.

If the UNet learned anything sample-specific, its denoising error on training
images must be lower than on held-out images from the same distribution.
The GMM posterior-mean denoiser (a perfect memorizer) calibrates what a real
memorization gap looks like in this metric.
"""
import sys, os, math, time
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, "/Users/ishittaiyer/Desktop/Research/src")
from multiband_data_utils import generate_multiband_dataset_postmask
from edm import EDMPrecond

torch.manual_seed(0)
device = "cpu"

# ── SmallUNet (identical to the notebooks) ──────────────────────────────────
class SinusoidalEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    def forward(self, t):
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, dtype=torch.float32, device=t.device) / (half - 1))
        args = t.float()[:, None] * freqs[None]
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)

class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, emb_dim):
        super().__init__()
        def ngroups(ch):
            for g in [8, 4, 2, 1]:
                if ch % g == 0: return g
        self.norm1 = nn.GroupNorm(ngroups(in_ch), in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(ngroups(out_ch), out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.emb_proj = nn.Linear(emb_dim, out_ch)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
    def forward(self, x, emb):
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.emb_proj(F.silu(emb))[:, :, None, None]
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)

class SmallUNet(nn.Module):
    def __init__(self, base_channels=16, emb_dim=64):
        super().__init__()
        C, E = base_channels, emb_dim
        self.time_embed = nn.Sequential(SinusoidalEmbedding(E), nn.Linear(E, E*2), nn.SiLU(), nn.Linear(E*2, E))
        self.conv_in = nn.Conv2d(1, C, 3, padding=1)
        self.enc1 = ResBlock(C, C, E)
        self.down1 = nn.Conv2d(C, C*2, 3, stride=2, padding=1)
        self.enc2 = ResBlock(C*2, C*2, E)
        self.down2 = nn.Conv2d(C*2, C*4, 3, stride=2, padding=1)
        self.mid = ResBlock(C*4, C*4, E)
        self.up2 = nn.ConvTranspose2d(C*4, C*2, 2, stride=2)
        self.dec2 = ResBlock(C*4, C*2, E)
        self.up1 = nn.ConvTranspose2d(C*2, C, 2, stride=2)
        self.dec1 = ResBlock(C*2, C, E)
        self.conv_out = nn.Conv2d(C, 1, 3, padding=1)
    def forward(self, x, t):
        emb = self.time_embed(t)
        h = self.conv_in(x)
        h1 = self.enc1(h, emb)
        h2 = self.enc2(self.down1(h1), emb)
        hm = self.mid(self.down2(h2), emb)
        hu = self.dec2(torch.cat([self.up2(hm), h2], dim=1), emb)
        hu = self.dec1(torch.cat([self.up1(hu), h1], dim=1), emb)
        return self.conv_out(hu)

# ── Data and checkpoints ────────────────────────────────────────────────────
components = [
    {"name": "coarse", "length_scale": 2.0,  "s": 2.0, "sigma_sq": 1.0, "band": (0.5, 4.0)},
    {"name": "mid1",   "length_scale": 6.0,  "s": 2.0, "sigma_sq": 1.0, "band": (4.0, 10.0)},
    {"name": "mid2",   "length_scale": 12.0, "s": 2.0, "sigma_sq": 1.0, "band": (10.0, 18.0)},
    {"name": "fine",   "length_scale": 24.0, "s": 2.0, "sigma_sq": 1.0, "band": (18.0, 32.0)},
]
result = generate_multiband_dataset_postmask(num_samples=200, grid_size=128, components=components,
                                             weights=[1.0, 0.8, 0.8, 1.2], seed=42, normalize=True)
x_all = result["combined"]
N = 128
x_held = x_all[100:116]          # 16 held-out fields, never seen in any training run

ckpts = torch.load("/Users/ishittaiyer/Desktop/Research/results/data/edm_unet_ntrain_checkpoints.pt",
                   map_location=device, weights_only=False)

SIGMAS = [0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]
K_NOISE = 4      # noise draws per image per sigma
CHUNK = 16

@torch.no_grad()
def unet_denoise_err(precond, imgs, sigma):
    """Mean per-image denoising MSE of D_theta(y + sigma*eps; sigma) vs y."""
    errs = []
    for k in range(K_NOISE):
        g = torch.Generator().manual_seed(1000 * k + int(sigma * 100))
        noise = torch.randn(imgs.shape, generator=g)
        x_noisy = imgs + sigma * noise
        for j in range(0, imgs.shape[0], CHUNK):
            xb = x_noisy[j:j+CHUNK].unsqueeze(1)
            sb = torch.full((xb.shape[0],), sigma)
            D = precond(xb, sb).squeeze(1)
            errs.append(((D - imgs[j:j+CHUNK]) ** 2).mean(dim=(-2, -1)))
    return torch.cat(errs).mean().item()

@torch.no_grad()
def gmm_denoise_err(train_flat, imgs, sigma):
    """Same metric for the GMM posterior-mean denoiser (perfect memorizer)."""
    errs = []
    for k in range(K_NOISE):
        g = torch.Generator().manual_seed(1000 * k + int(sigma * 100))
        noise = torch.randn(imgs.shape, generator=g)
        x_noisy = (imgs + sigma * noise).reshape(imgs.shape[0], -1)
        d2 = torch.cdist(x_noisy, train_flat) ** 2
        w = torch.softmax(-d2 / (2 * sigma ** 2), dim=1)
        D = (w @ train_flat).reshape(imgs.shape)
        errs.append(((D - imgs) ** 2).mean(dim=(-2, -1)))
    return torch.cat(errs).mean().item()

results = {}
for n_train, entry in ckpts.items():
    unet = SmallUNet(base_channels=16, emb_dim=64)
    precond = EDMPrecond(unet, sigma_data=entry["sigma_data"])
    precond.load_state_dict(entry["state_dict"])
    precond.eval()

    x_train = x_all[:n_train]
    train_flat = x_train.reshape(n_train, -1)

    rows = []
    for sigma in SIGMAS:
        t0 = time.time()
        u_tr = unet_denoise_err(precond, x_train, sigma)
        u_ho = unet_denoise_err(precond, x_held, sigma)
        g_tr = gmm_denoise_err(train_flat, x_train, sigma)
        g_ho = gmm_denoise_err(train_flat, x_held, sigma)
        rows.append(dict(sigma=sigma, unet_train=u_tr, unet_held=u_ho,
                         gmm_train=g_tr, gmm_held=g_ho))
        print(f"n={n_train:>2d} sigma={sigma:>5.2f} | UNet train={u_tr:.5f} held={u_ho:.5f} "
              f"gap={u_ho/max(u_tr,1e-12):>6.2f}x | GMM train={g_tr:.5f} held={g_ho:.5f} "
              f"gap={g_ho/max(g_tr,1e-12):>8.1f}x  ({time.time()-t0:.0f}s)", flush=True)
    results[n_train] = rows
    print()

torch.save({"results": results, "sigmas": SIGMAS, "k_noise": K_NOISE,
            "held_idx": [100, 116],
            "note": ("Train vs held-out denoising MSE of the saved EDM UNet checkpoints and the "
                     "GMM posterior-mean denoiser (memorization calibrator). gap = held/train; "
                     "gap ~ 1 for the UNet means nothing sample-specific was learned.")},
           "/Users/ishittaiyer/Desktop/Research/results/data/edm_unet_denoiser_gap.pt")
print("saved -> results/data/edm_unet_denoiser_gap.pt")
