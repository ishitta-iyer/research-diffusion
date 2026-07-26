"""Train/held-out denoiser gap test on the saved EDM UNet checkpoints.

If the UNet learned anything sample-specific, its denoising error on training
images must be lower than on held-out images from the same distribution.
The GMM posterior-mean denoiser (a perfect memorizer) calibrates what a real
memorization gap looks like in this metric.
"""
import sys, os, math, time
from pathlib import Path
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
from multiband_data_utils import generate_multiband_dataset_postmask
from edm import EDMPrecond
from unet import SmallUNet, remap_legacy_state_dict

DATA_DIR = REPO_ROOT / "results" / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
torch.manual_seed(0)
device = "cpu"

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

ckpts = torch.load(DATA_DIR / "edm_unet_ntrain_checkpoints.pt",
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
    try:
        precond.load_state_dict(entry["state_dict"])
    except RuntimeError:
        precond.load_state_dict(remap_legacy_state_dict(entry["state_dict"]))
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
           DATA_DIR / "edm_unet_denoiser_gap.pt")
print("saved -> results/data/edm_unet_denoiser_gap.pt")
