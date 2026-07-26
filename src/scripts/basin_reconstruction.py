"""Basin-reachability test: run the reverse SDE from noised TRAINING images
y_i + sigma0*eps at varying sigma0, and check whether the sampler reconstructs
the source image. Trick: VE_EDM(sigma_max=sigma0) with latents=(y+sigma0*eps)/sigma0
makes SDEsampler start exactly at the forward marginal at sigma0.

Prediction from the denoiser-gap result: the UNet reconstructs for small sigma0
(memorized basins exist locally) and loses the source identity once sigma0 exceeds
the range where its train/held-out gap dies (~2-5); the GMM reconstructs from all
sigma0. Together with the gap test this shows: memorization is present in the
network, but pure-noise sampling makes its basin decision where the UNet has no
sample-specific knowledge.
"""
import sys, math, time
from pathlib import Path
import torch

torch.set_num_threads(2)   # leave CPU headroom for the CoordConv training job

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
import diffusion_score_models as score_models
from multiband_data_utils import generate_multiband_dataset_postmask
from edm import EDMPrecond, EDMScoreWrapper
from unet import SmallUNet, remap_legacy_state_dict

DATA_DIR = REPO_ROOT / "results" / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
device = "cpu"

# ── Data + checkpoint (n_train = 4) ─────────────────────────────────────────
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
N_TRAIN = 4
x_train = x_all[:N_TRAIN]
train_flat = x_train.reshape(N_TRAIN, -1)

ck = torch.load(DATA_DIR / "edm_unet_ntrain_checkpoints.pt",
                map_location=device, weights_only=False)[N_TRAIN]
unet = SmallUNet(16, 64)
precond = EDMPrecond(unet, sigma_data=ck["sigma_data"])
try:
    precond.load_state_dict(ck["state_dict"])
except RuntimeError:
    precond.load_state_dict(remap_legacy_state_dict(ck["state_dict"]))
precond.eval()

SIGMA0 = [0.3, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0]
K_NOISE = 4
STEPS_FULL = 1000        # steps for sigma0=10; scaled down for smaller sigma0


def reconstruct(score_builder, sigma0, n_steps):
    """Reverse SDE from y_i + sigma0*eps for all i, K_NOISE draws. Returns (B, N, N)."""
    ve = score_models.VE_EDM(sigma_min=0.002, sigma_max=sigma0)
    score_fn = score_builder(ve)
    torch.manual_seed(int(sigma0 * 1000))
    src = x_train.repeat_interleave(K_NOISE, dim=0)              # (16, N, N)
    noise = torch.randn_like(src)
    x0 = (src + sigma0 * noise).reshape(-1, N * N)
    latents = x0 / sigma0        # SDEsampler multiplies by marginal_prob_std(T)=sigma0
    out = ve.SDEsampler(score_fn, latents, num_steps=n_steps)
    return out.reshape(-1, N, N), src


def identity_metrics(x_rec, src_idx):
    """Relative L2 to the source vs to the best OTHER training image; identity accuracy."""
    flat = x_rec.reshape(x_rec.shape[0], -1)
    d = torch.cdist(flat, train_flat)                            # (B, n_train)
    norm = train_flat.norm(dim=1)[None, :]
    rel = d / norm
    src = rel[torch.arange(len(src_idx)), src_idx]
    other = rel.clone()
    other[torch.arange(len(src_idx)), src_idx] = float("inf")
    best_other = other.min(dim=1).values
    acc = (d.argmin(dim=1) == src_idx).float().mean().item()
    return src.mean().item(), best_other.mean().item(), acc


src_idx = torch.arange(N_TRAIN).repeat_interleave(K_NOISE)
rows = []
print(f"{'sigma0':>7} | {'UNet rel_src':>12} {'rel_other':>10} {'id_acc':>7} | "
      f"{'GMM rel_src':>11} {'rel_other':>10} {'id_acc':>7}", flush=True)
for sigma0 in SIGMA0:
    n_steps = max(300, int(STEPS_FULL * math.log(sigma0 / 0.002) / math.log(10.0 / 0.002)))
    t0 = time.time()

    unet_builder = lambda ve: EDMScoreWrapper(precond, ve.marginal_prob_std, N, c_tikhonov=0.0)
    xr_u, _ = reconstruct(unet_builder, sigma0, n_steps)
    u_src, u_oth, u_acc = identity_metrics(xr_u, src_idx)

    gmm_builder = lambda ve: score_models.GMM_score(train_flat, ve.marginal_prob_mean, ve.marginal_prob_std)
    xr_g, _ = reconstruct(gmm_builder, sigma0, n_steps)
    g_src, g_oth, g_acc = identity_metrics(xr_g, src_idx)

    rows.append(dict(sigma0=sigma0, n_steps=n_steps,
                     unet_rel_src=u_src, unet_rel_other=u_oth, unet_id_acc=u_acc,
                     gmm_rel_src=g_src, gmm_rel_other=g_oth, gmm_id_acc=g_acc,
                     unet_recs=xr_u[:4].cpu()))
    print(f"{sigma0:>7.1f} | {u_src:>12.4f} {u_oth:>10.4f} {u_acc:>7.2f} | "
          f"{g_src:>11.4f} {g_oth:>10.4f} {g_acc:>7.2f}   ({time.time()-t0:.0f}s, {n_steps} steps)", flush=True)

torch.save({"rows": rows, "sigma0_values": SIGMA0, "n_train": N_TRAIN, "k_noise": K_NOISE,
            "x_train": x_train.cpu(),
            "note": ("Basin-reachability: reverse SDE started from y_i + sigma0*eps (n_train=4 "
                     "checkpoint). rel_src = relative L2 of the reconstruction to its source "
                     "image; rel_other = to the best other training image; id_acc = fraction "
                     "where the source is the nearest training image. GMM same protocol.")},
           DATA_DIR / "edm_unet_basin_reconstruction.pt")
print("saved -> results/data/edm_unet_basin_reconstruction.pt")
