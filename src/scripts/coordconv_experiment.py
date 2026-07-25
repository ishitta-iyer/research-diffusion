"""CoordConv ablation: give the EDM UNet coordinate channels so it CAN break
translation equivariance, retrain at small n_train, and test whether memorization
appears under the fixed sampler (sigma_max=10, 1000 SDE steps).

If memorization appears only with coordinates, the architectural inductive-bias
explanation for the null result is demonstrated constructively.
"""
import sys, os, math, time
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, "/Users/ishittaiyer/Desktop/Research/src")
import diffusion_score_models as score_models
from multiband_data_utils import (generate_multiband_dataset_postmask, make_radial_band_mask,
                                  radial_bandpass, make_radial_k_grid)
from edm import EDMPrecond, EDMScoreWrapper, train_edm

device = "cpu"
torch.manual_seed(0)

# ── UNet blocks (identical to the notebooks) ────────────────────────────────
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

class CoordSmallUNet(nn.Module):
    """SmallUNet + two coordinate channels at the input (CoordConv).

    The ONLY change vs SmallUNet: conv_in takes 3 channels (image, x, y),
    with coordinates in [-1, 1]. This gives the network absolute position —
    the one thing a translation-equivariant conv net cannot represent —
    so it can in principle store and reproduce specific training fields.
    """
    def __init__(self, base_channels=16, emb_dim=64):
        super().__init__()
        C, E = base_channels, emb_dim
        self.time_embed = nn.Sequential(SinusoidalEmbedding(E), nn.Linear(E, E*2), nn.SiLU(), nn.Linear(E*2, E))
        self.conv_in = nn.Conv2d(3, C, 3, padding=1)   # <- 3 channels instead of 1
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
        self.register_buffer("coords", self._make_coords(128), persistent=False)
    @staticmethod
    def _make_coords(N):
        r = torch.linspace(-1.0, 1.0, N)
        yy, xx = torch.meshgrid(r, r, indexing="ij")
        return torch.stack([xx, yy], dim=0)   # (2, N, N)
    def forward(self, x, t):
        B = x.shape[0]
        x = torch.cat([x, self.coords.unsqueeze(0).expand(B, -1, -1, -1)], dim=1)
        emb = self.time_embed(t)
        h = self.conv_in(x)
        h1 = self.enc1(h, emb)
        h2 = self.enc2(self.down1(h1), emb)
        hm = self.mid(self.down2(h2), emb)
        hu = self.dec2(torch.cat([self.up2(hm), h2], dim=1), emb)
        hu = self.dec1(torch.cat([self.up1(hu), h1], dim=1), emb)
        return self.conv_out(hu)

# ── Data ────────────────────────────────────────────────────────────────────
components = [
    {"name": "coarse", "length_scale": 2.0,  "s": 2.0, "sigma_sq": 1.0, "band": (0.5, 4.0)},
    {"name": "mid1",   "length_scale": 6.0,  "s": 2.0, "sigma_sq": 1.0, "band": (4.0, 10.0)},
    {"name": "mid2",   "length_scale": 12.0, "s": 2.0, "sigma_sq": 1.0, "band": (10.0, 18.0)},
    {"name": "fine",   "length_scale": 24.0, "s": 2.0, "sigma_sq": 1.0, "band": (18.0, 32.0)},
]
result = generate_multiband_dataset_postmask(num_samples=200, grid_size=128, components=components,
                                             weights=[1.0, 0.8, 0.8, 1.2], seed=42, normalize=True)
bands = result["bands"]
x_all = result["combined"]
N = 128

# ── Metrics (identical to the notebooks) ────────────────────────────────────
@torch.no_grad()
def band_component(x, band, norm="forward"):
    if x.dim() == 2:
        x = x.unsqueeze(0)
    k_lo, k_hi = band
    m = make_radial_band_mask(x.shape[-1], k_lo, k_hi, device=x.device, dtype=x.dtype)
    return radial_bandpass(x, m, norm=norm)

@torch.no_grad()
def nn_by_coarse(x_gen, x_train, coarse_band):
    cg = band_component(x_gen, coarse_band).flatten(1)
    ct = band_component(x_train, coarse_band).flatten(1)
    D = torch.cdist(cg, ct)
    return D.argmin(dim=1), D.min(dim=1).values

def make_ring_masks(N, k_edges):
    kr = make_radial_k_grid(N)
    return torch.stack([((kr >= lo) & (kr < hi)).float() for lo, hi in zip(k_edges[:-1], k_edges[1:])])

@torch.no_grad()
def ring_rel_l2(xa, xb, ring_masks, eps=1e-12):
    Xa = torch.fft.fft2(xa, dim=(-2, -1))
    Xb = torch.fft.fft2(xb, dim=(-2, -1))
    D = Xa - Xb
    rm = ring_masks.unsqueeze(0)
    num = ((D.abs() ** 2).unsqueeze(1) * rm).sum(dim=(-2, -1))
    den = ((Xa.abs() ** 2).unsqueeze(1) * rm).sum(dim=(-2, -1)).clamp_min(eps)
    return torch.sqrt(num / den)

@torch.no_grad()
def random_baseline_errors(xg, xtr, ring_masks, n_ref=32):
    errs = []
    for _ in range(n_ref):
        idx = torch.randint(0, xtr.shape[0], (xg.shape[0],))
        errs.append(ring_rel_l2(xg, xtr[idx], ring_masks))
    return torch.stack(errs, dim=0)

kmax = float(make_radial_k_grid(N).max().item())
k_edges = torch.arange(0.5, kmax + 1.5, 1.0)
ring_masks = make_ring_masks(N, k_edges)
k_centers = 0.5 * (k_edges[1:] + k_edges[:-1])
def ring_idx(band):
    lo, hi = band
    return ((k_centers >= lo) & (k_centers < hi)).nonzero().squeeze(1)
coarse_r, fine_r = ring_idx(bands["coarse"]), ring_idx(bands["fine"])

def memorization_metrics(x_gen, xtr):
    idx_nn, _ = nn_by_coarse(x_gen, xtr, bands["coarse"])
    err_nn = ring_rel_l2(x_gen, xtr[idx_nn], ring_masks)
    err_rand = random_baseline_errors(x_gen, xtr, ring_masks, n_ref=32)
    ratio = (err_nn.unsqueeze(0) / (err_rand + 1e-12)).mean(dim=0)
    return {"ratio": ratio, "mean_ratio": ratio.mean(dim=0),
            "coarse_score": ratio[:, coarse_r].mean(dim=1),
            "fine_score": ratio[:, fine_r].mean(dim=1),
            "idx_nn": idx_nn}

# ── Train + evaluate ─────────────────────────────────────────────────────────
N_TRAIN_SWEEP = [4, 8]
TOTAL_STEPS = 5000
N_GEN, N_SDE_STEPS = 16, 1000    # config D sampler
ve = score_models.VE_EDM(sigma_min=0.002, sigma_max=10.0)

out = {}
for n_train in N_TRAIN_SWEEP:
    print(f"\n===== CoordConv n_train = {n_train} =====", flush=True)
    x_train = x_all[:n_train]
    train_flat = x_train.reshape(n_train, -1).contiguous()

    t0 = time.time()
    ckpts = train_edm(train_flat, grid_size=N, total_steps=TOTAL_STEPS,
                      checkpoint_at=[TOTAL_STEPS], base_channels=16, emb_dim=64,
                      lr=1e-3, batch_size=n_train, P_mean=-1.2, P_std=1.2,
                      seed=0, device=device, UNetClass=CoordSmallUNet)
    precond = ckpts[TOTAL_STEPS]
    print(f"  training took {(time.time()-t0)/60:.1f} min", flush=True)

    wrapper = EDMScoreWrapper(precond, ve.marginal_prob_std, N, c_tikhonov=0.0)
    torch.manual_seed(42)
    latents = torch.randn(N_GEN, N * N)
    t0 = time.time()
    samples = ve.SDEsampler(wrapper, latents, num_steps=N_SDE_STEPS)
    x_gen = samples.reshape(N_GEN, N, N)
    print(f"  sampling took {(time.time()-t0)/60:.1f} min", flush=True)

    m = memorization_metrics(x_gen, x_train)
    m["samples"] = x_gen.cpu()
    m["state_dict"] = precond.state_dict()
    m["sigma_data"] = precond.sigma_data
    out[n_train] = m
    print(f"  CoordConv UNet: coarse={m['coarse_score'].mean():.4f}  "
          f"fine={m['fine_score'].mean():.4f}", flush=True)
    print(f"  NN match counts: {torch.bincount(m['idx_nn'], minlength=n_train).tolist()}", flush=True)

torch.save({"results": out, "n_train_values": N_TRAIN_SWEEP, "k_centers": k_centers,
            "bands": bands, "sampler": {"sigma_max": 10.0, "n_steps": N_SDE_STEPS},
            "note": ("CoordConv EDM UNet (SmallUNet + 2 coordinate input channels), trained "
                     "identically to the baseline (5000 full-batch steps, seed 0), sampled with "
                     "the fixed sampler (sigma_max=10, 1000 steps, latents seed 42). Baseline "
                     "comparison: config D in edm_unet_sigma_fix_ablation.pt.")},
           "/Users/ishittaiyer/Desktop/Research/results/data/edm_unet_coordconv.pt")
print("\nsaved -> results/data/edm_unet_coordconv.pt")
