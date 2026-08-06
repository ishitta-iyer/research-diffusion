"""Per-wavenumber memorization metric, consolidated from the analysis notebooks
(analyze_coarse_fine_memorization.py / edm_tikhonov_memorization.ipynb /
gmm_covariance_tikhonov_memorization.ipynb).

Pipeline: for each generated sample, find its nearest training neighbour in the
coarse band, then compute the per-ring relative L2 error in Fourier space to that
neighbour, normalized by the same error to random training images:

    ratio(k) = err_to_NN(k) / err_to_random(k)

ratio < 1  =>  the generated sample is closer to its NN than to random training
images in that band (memorized); ratio ~ 1 => novel in that band.

Two conventions are baked in, and paper/main.tex states both:

* Normalization. `ring_rel_l2` normalizes *both* terms by the **generated
  sample's** ring norm, so the normalization cancels in the ratio and the metric
  is a pure error ratio. This is deliberate.
* Aggregation. The ratio is the mean over the J reference draws of the per-draw
  ratio (mean-of-ratios). Every committed result uses this.
"""

import torch

from multiband_data_utils import make_radial_band_mask, radial_bandpass, make_radial_k_grid


@torch.no_grad()
def band_component(x, band, norm='forward'):
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
    idx = D.argmin(dim=1)
    return idx, D.min(dim=1).values


def make_ring_masks(N, k_edges, device='cpu'):
    kr = make_radial_k_grid(N, device=device)
    masks = []
    for lo, hi in zip(k_edges[:-1], k_edges[1:]):
        masks.append(((kr >= lo) & (kr < hi)).float())
    return torch.stack(masks, dim=0)


@torch.no_grad()
def ring_rel_l2(xa, xb, ring_masks, fft_norm=None, eps=1e-12):
    """Per-ring relative L2 distance between xa and xb, normalized by **xa**.

    Call sites pass xa = x_gen, so both the NN error and the random-reference
    errors are divided by the same quantity, the generated sample's ring norm.
    The normalization therefore cancels out of the memorization ratio, which is a
    pure error ratio. This is intentional and is what paper/main.tex states;
    normalizing each term by its own reference field instead would not cancel
    (measured effect on held-out fields: +0.073 on the coarse band).
    """
    Xa = torch.fft.fft2(xa, dim=(-2, -1), norm=fft_norm)
    Xb = torch.fft.fft2(xb, dim=(-2, -1), norm=fft_norm)
    D = Xa - Xb
    rm = ring_masks.unsqueeze(0).to(Xa.device)
    num = (D.abs() ** 2).unsqueeze(1) * rm
    den = (Xa.abs() ** 2).unsqueeze(1) * rm
    num = num.sum(dim=(-2, -1))
    den = den.sum(dim=(-2, -1)).clamp_min(eps)
    return torch.sqrt(num / den)


@torch.no_grad()
def random_baseline_errors(xg, xtr, ring_masks, n_ref=32, fft_norm=None, exclude_idx=None):
    """Reference errors to randomly drawn training fields.

    exclude_idx: optional (G,) tensor of per-sample indices to exclude from the draws
    (normally the nearest neighbour). Without it the NN is itself part of the baseline,
    which floors the ratio at ~1/n_train and makes scores incomparable across training
    set sizes: with n_train=2 a random draw IS the NN half the time, so even perfect
    memorization gives ratio ~0.5. Excluding it removes that floor.

    It does NOT by itself make the metric n_train-independent, and no choice of
    aggregation makes it so either. Two residual n_train dependencies remain:

    1. A Jensen gap under the default mean-of-ratios aggregation, ~0.03 on the coarse
       band. It is exactly 0 at n_train=2 (only one non-NN reference exists, so
       err_rand has zero variance across draws) and ~0.03 from n_train=4 up.
    2. A larger confound that no aggregation touches: the nearest neighbour is an
       argmin over n_train candidates, so the best-of-n_train match improves as
       n_train grows and the neutral baseline itself falls.

    The remedy for both is the **neutral baseline**, not the aggregation knob: report
    every score against the per-n_train score of held-out real fields (perfect
    generalization), computed under the same convention. Measured coarse-band neutral
    baseline, exclude_nn=True, mean-of-ratios:

        n_train:  2       4       8       16      32
                  0.8567  0.9050  0.8448  0.8150  0.8126
    """
    n_train = xtr.shape[0]
    errs = []
    for _ in range(n_ref):
        rand_idx = torch.randint(0, n_train, (xg.shape[0],), device=xtr.device)
        if exclude_idx is not None and n_train > 1:
            # resample collisions onto a different index (uniform over the other n-1)
            clash = rand_idx == exclude_idx
            if clash.any():
                shift = torch.randint(1, n_train, (int(clash.sum()),), device=xtr.device)
                rand_idx[clash] = (exclude_idx[clash] + shift) % n_train
        errs.append(ring_rel_l2(xg, xtr[rand_idx], ring_masks, fft_norm=fft_norm))
    return torch.stack(errs, dim=0)  # (n_ref, G, K)


def ring_idx_for_band(band, k_centers):
    lo, hi = band
    return ((k_centers >= lo) & (k_centers < hi)).nonzero().squeeze(1)


@torch.no_grad()
def radial_power_spectrum(x, ring_masks_):
    """Mean per-mode power in each wavenumber ring (orthonormal FFT), averaged over batch."""
    X = torch.fft.fft2(x, dim=(-2, -1), norm='ortho')
    P = X.abs() ** 2
    rm = ring_masks_.to(x.device)
    num = (P.unsqueeze(1) * rm.unsqueeze(0)).sum(dim=(-2, -1))   # (G, K)
    cnt = rm.sum(dim=(-2, -1)).clamp_min(1)
    return (num / cnt).mean(dim=0)                               # (K,)


class RingMetricContext:
    """Precomputed ring masks + band ring indices for a grid size, plus the
    standard memorization evaluation used across the sweep notebooks."""

    def __init__(self, N, bands, device='cpu'):
        self.N = N
        self.bands = bands
        kmax = float(make_radial_k_grid(N, device=device).max().item())
        self.k_edges = torch.arange(0.5, kmax + 1.5, 1.0, device=device)
        self.ring_masks = make_ring_masks(N, self.k_edges, device=device)
        self.k_centers = 0.5 * (self.k_edges[1:] + self.k_edges[:-1])
        self.band_rings = {name: ring_idx_for_band(band, self.k_centers)
                           for name, band in bands.items()}

    @torch.no_grad()
    def evaluate(self, x_gen, x_train, n_rand_ref=32, exclude_nn=False,
                 aggregate='mean_of_ratios'):
        """Returns dict with per-k mean ratio, per-sample coarse/fine scores, and NN indices.

        exclude_nn=True drops the nearest neighbour from the random baseline, which is
        required whenever scores are compared across different training-set sizes
        (see random_baseline_errors). Default False preserves the metric used by the
        earlier committed sweeps, which all run at a single n_train.

        aggregate: 'mean_of_ratios' (default) computes mean_j(e_NN / e_rand_j). This is
        the definition stated in paper/main.tex eq. (5)-(6) and the one every committed
        result uses. 'ratio_of_means' computes e_NN / mean_j(e_rand_j) and is available
        as an alternative aggregation. By Jensen, mean-of-ratios >= ratio-of-means, by an
        amount proportional to the variance of e_rand_j across draws.

        On an n_train axis, do not switch aggregation to chase comparability — neither
        choice removes the n_train dependence (see random_baseline_errors). Report scores
        against the per-n_train neutral baseline computed under the same convention.
        n_train=2 is the zero-variance special case: with exclude_nn=True only one
        reference remains, so the two aggregations coincide exactly.
        """
        idx_nn, _ = nn_by_coarse(x_gen, x_train, self.bands['coarse'])
        err_nn = ring_rel_l2(x_gen, x_train[idx_nn], self.ring_masks)
        err_rand = random_baseline_errors(x_gen, x_train, self.ring_masks, n_ref=n_rand_ref,
                                          exclude_idx=idx_nn if exclude_nn else None)
        if aggregate == 'mean_of_ratios':
            ratio = (err_nn.unsqueeze(0) / (err_rand + 1e-12)).mean(dim=0)   # (G, K)
        elif aggregate == 'ratio_of_means':
            ratio = err_nn / (err_rand.mean(dim=0) + 1e-12)                  # (G, K)
        else:
            raise ValueError(f"unknown aggregate {aggregate!r}")
        out = {
            'ratio': ratio,
            'mean_ratio': ratio.mean(dim=0),
            'idx_nn': idx_nn,
        }
        for name, rings in self.band_rings.items():
            out[f'{name}_score'] = ratio[:, rings].mean(dim=1)
        return out
