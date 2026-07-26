"""Checks for the consolidated src modules (run on CPU for exactness):

1. GMM_score_CovarianceTikhonov reduces to GMM_score at c=0 and to
   GMM_score_TikhonovRegularized at lambda == 1 (both 'inverse' and 'direct').
2. src/unet.py SmallUNet(16, 64, num_levels=3) is layer-for-layer identical to the
   legacy inline SmallUNet: same creation order and shapes, so same init under a
   fixed seed and identical forward outputs.
3. remap_legacy_state_dict loads a legacy state_dict into the consolidated net.

Run: python3 src/tests/test_consolidation.py   (or pytest src/tests/)
"""
import sys, math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import diffusion_score_models as score_models
from unet import SmallUNet, SinusoidalEmbedding, ResBlock, count_parameters, remap_legacy_state_dict
from device_utils import resolve_device


# Legacy inline SmallUNet, verbatim from the pre-consolidation notebooks/scripts,
# kept here only as the reference for the equivalence test.
class _LegacySmallUNet(nn.Module):
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


def _toy_gmm_setup(n_train=8, N=16, seed=0):
    torch.manual_seed(seed)
    train_flat = torch.randn(n_train, N * N)
    ve = score_models.VE_EDM(sigma_min=0.002, sigma_max=80.0)
    x = torch.randn(4, N * N)
    t = torch.rand(4) * 0.6 + 0.2
    return train_flat, ve, x, t, N


def test_covariance_reduces_to_gmm_at_c0():
    train_flat, ve, x, t, N = _toy_gmm_setup()
    s_cov = score_models.GMM_score_CovarianceTikhonov(
        train_flat, ve.marginal_prob_mean, ve.marginal_prob_std, N, constant=0.0)(x, t)
    s_gmm = score_models.GMM_score(train_flat, ve.marginal_prob_mean, ve.marginal_prob_std)(x, t)
    rel = ((s_cov - s_gmm).abs().max() / s_gmm.abs().max()).item()
    assert rel < 1e-4, f"c=0 covariance score differs from GMM score: rel={rel:.2e}"


def test_covariance_reduces_to_isotropic_at_unit_spectrum():
    train_flat, ve, x, t, N = _toy_gmm_setup()
    ones = torch.ones(N, N)
    s_tik = score_models.GMM_score_TikhonovRegularized(
        train_flat, ve.marginal_prob_mean, ve.marginal_prob_std,
        ve.diffusion_coeff, constant=0.37)(x, t)
    for weighting in ("inverse", "direct"):
        s_cov = score_models.GMM_score_CovarianceTikhonov(
            train_flat, ve.marginal_prob_mean, ve.marginal_prob_std, N,
            constant=0.37, spectrum=ones, weighting=weighting)(x, t)
        rel = ((s_cov - s_tik).abs().max() / s_tik.abs().max()).item()
        assert rel < 1e-4, f"lambda==1 ({weighting}) differs from isotropic Tikhonov: rel={rel:.2e}"


def test_unet_matches_legacy_definition():
    torch.manual_seed(123)
    legacy = _LegacySmallUNet(16, 64)
    torch.manual_seed(123)
    new = SmallUNet(base_channels=16, emb_dim=64, num_levels=3)
    n_legacy, n_new = count_parameters(legacy), count_parameters(new)
    assert n_legacy == n_new, f"param count changed: legacy {n_legacy} vs new {n_new}"
    x = torch.randn(2, 1, 32, 32)
    t = torch.randn(2)
    with torch.no_grad():
        d = (legacy(x, t) - new(x, t)).abs().max().item()
    assert d == 0.0, f"forward outputs differ (max abs {d:.3e}) — layer order/shape changed"


def test_remap_legacy_state_dict():
    legacy = _LegacySmallUNet(16, 64)
    new = SmallUNet(base_channels=16, emb_dim=64, num_levels=3)
    new.load_state_dict(remap_legacy_state_dict(legacy.state_dict()))
    x = torch.randn(2, 1, 32, 32)
    t = torch.randn(2)
    with torch.no_grad():
        d = (legacy(x, t) - new(x, t)).abs().max().item()
    assert d == 0.0, f"remapped state_dict gives different outputs (max abs {d:.3e})"


def test_num_levels_variants_run():
    for L in (2, 3, 4):
        net = SmallUNet(base_channels=8, emb_dim=32, num_levels=L)
        x = torch.randn(2, 1, 64, 64)
        t = torch.randn(2)
        with torch.no_grad():
            y = net(x, t)
        assert y.shape == x.shape, f"num_levels={L}: output shape {y.shape} != input {x.shape}"


def _fit_free_denoiser(x0, x_noisy, sigma, **pen_kwargs):
    """Minimize the train_edm objective over a FREE denoiser output D (no network), so the
    optimum is the analytic stationary point. Returns the implied score (D - x)/sigma^2."""
    from edm import tikhonov_penalty
    D = x_noisy.clone().requires_grad_(True)
    steps = 4000
    opt = torch.optim.Adam([D], lr=0.05)
    # cosine decay: without it Adam's noise floor leaves ~1e-2 relative error, which is
    # large enough to mask a genuine formula error
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    for _ in range(steps):
        opt.zero_grad()
        loss = ((D - x0) ** 2).mean(dim=(1, 2, 3))
        loss = loss + tikhonov_penalty(D, x_noisy, sigma, **pen_kwargs)
        loss.sum().backward()
        opt.step()
        sched.step()
    assert D.grad.abs().max().item() < 1e-5, "free-D optimization did not converge"
    return ((D.detach() - x_noisy) / sigma[:, None, None, None] ** 2)


def test_isotropic_penalty_stationary_point():
    """Minimizer of ||D-x0||^2 + c||D-x||^2/sigma^2 must give s* = (x0-x)/(sigma^2+c),
    i.e. Baptista et al. eq. (5.7) with Gamma = (c/sigma^2) I."""
    torch.manual_seed(0)
    N, c = 8, 0.3
    x0 = torch.randn(2, 1, N, N)
    sigma = torch.tensor([0.7, 1.3])
    x_noisy = x0 + sigma[:, None, None, None] * torch.randn_like(x0)
    got = _fit_free_denoiser(x0, x_noisy, sigma, c_tikhonov=c)
    expect = (x0 - x_noisy) / (sigma[:, None, None, None] ** 2 + c)
    rel = ((got - expect).abs().max() / expect.abs().max()).item()
    assert rel < 1e-3, f"isotropic stationary score off by rel={rel:.2e}"


def test_covariance_penalty_stationary_point():
    """Minimizer with the covariance penalty must give, per Fourier mode,
    s*_k = (x0-x)_k/(sigma^2 + c/lambda(k)) — the training-time analogue of
    GMM_score_CovarianceTikhonov."""
    from edm import hermitian_symmetrize
    torch.manual_seed(0)
    N, c = 8, 0.3
    # a real power spectrum, i.e. Hermitian-symmetric (lambda_k = lambda_-k). An arbitrary
    # asymmetric lambda has no real-valued minimizer and the per-mode formula would not hold.
    lam = (torch.fft.fft2(torch.randn(64, N, N), norm='ortho').abs() ** 2).mean(0) + 0.05
    assert torch.allclose(lam, hermitian_symmetrize(lam), atol=1e-6)
    inv_lam = (1.0 / lam)[None, None, :, :]
    x0 = torch.randn(2, 1, N, N)
    sigma = torch.tensor([0.7, 1.3])
    x_noisy = x0 + sigma[:, None, None, None] * torch.randn_like(x0)

    got = _fit_free_denoiser(x0, x_noisy, sigma, c_tikhonov_cov=c, inv_lam=inv_lam)

    R = torch.fft.fft2(x0 - x_noisy, dim=(-2, -1), norm='ortho')
    denom = sigma[:, None, None, None] ** 2 + c * inv_lam
    expect = torch.fft.ifft2(R / denom, dim=(-2, -1), norm='ortho').real
    rel = ((got - expect).abs().max() / expect.abs().max()).item()
    assert rel < 1e-3, f"covariance stationary score off by rel={rel:.2e}"


def test_hermitian_symmetrize():
    """No-op on real-data spectra; a genuine projection on asymmetric input."""
    from edm import hermitian_symmetrize
    torch.manual_seed(0)
    real_spec = (torch.fft.fft2(torch.randn(32, 8, 8), norm='ortho').abs() ** 2).mean(0)
    assert torch.allclose(real_spec, hermitian_symmetrize(real_spec), atol=1e-6), \
        "symmetrization must not alter a spectrum estimated from real data"
    asym = torch.rand(8, 8) + 0.1
    sym = hermitian_symmetrize(asym)
    assert not torch.allclose(asym, sym, atol=1e-3), "asymmetric input should be changed"
    assert torch.allclose(sym, hermitian_symmetrize(sym), atol=1e-6), "must be idempotent"


def test_covariance_penalty_reduces_to_isotropic_at_unit_spectrum():
    """lambda == 1 must make the covariance penalty numerically identical to the isotropic
    one (Parseval), so the two knobs agree in the limit."""
    from edm import tikhonov_penalty
    torch.manual_seed(0)
    N, c = 8, 0.3
    D = torch.randn(3, 1, N, N)
    x = torch.randn(3, 1, N, N)
    sigma = torch.tensor([0.5, 1.0, 2.0])
    iso = tikhonov_penalty(D, x, sigma, c_tikhonov=c)
    cov = tikhonov_penalty(D, x, sigma, c_tikhonov_cov=c,
                           inv_lam=torch.ones(1, 1, N, N))
    rel = ((iso - cov).abs().max() / iso.abs().max()).item()
    assert rel < 1e-5, f"lambda==1 penalty differs from isotropic: rel={rel:.2e}"


def test_train_edm_unregularized_path_unchanged():
    """c=0 must leave the loss byte-identical to the pre-Tikhonov code path."""
    from edm import train_edm, EDMPrecond
    torch.manual_seed(0)
    data = torch.randn(4, 16 * 16)
    kw = dict(grid_size=16, total_steps=3, checkpoint_at=[3], base_channels=8,
              emb_dim=32, batch_size=4, seed=0, device="cpu", UNetClass=SmallUNet)
    a = train_edm(data, **kw)[3]
    b = train_edm(data, c_tikhonov=0.0, c_tikhonov_cov=0.0, **kw)[3]
    for (k, va), (_, vb) in zip(a.state_dict().items(), b.state_dict().items()):
        assert torch.equal(va, vb), f"c=0 path changed weights at {k}"


def test_resolve_device():
    assert resolve_device("cpu").type == "cpu"
    d = resolve_device()
    assert d.type in ("cpu", "mps", "cuda")


def test_exclude_nn_makes_ratio_comparable_across_n_train():
    """Perfect memorization (generated == a training field) must score ~0 regardless of
    n_train. Without exclude_nn the random baseline contains the NN itself, flooring the
    ratio near 1/n_train (~0.5 at n_train=2), which would corrupt any across-n_train
    comparison in the transition/capacity sweeps."""
    from memorization_metrics import RingMetricContext
    N = 32
    bands = {'coarse': (0.5, 4.0), 'fine': (8.0, 16.0)}
    torch.manual_seed(0)
    pool = torch.randn(8, N, N)
    ctx = RingMetricContext(N, bands, device='cpu')
    for n_train in (2, 4, 8):
        x_train = pool[:n_train]
        # near-memorization: a training field plus a small perturbation. (Exactly equal
        # fields give err_nn = 0, so the ratio is 0 whatever the denominator — the
        # inflation only shows up for the realistic near-miss case.)
        torch.manual_seed(1)
        x_gen = x_train + 0.01 * torch.randn_like(x_train)
        excl = ctx.evaluate(x_gen, x_train, n_rand_ref=64, exclude_nn=True)
        incl = ctx.evaluate(x_gen, x_train, n_rand_ref=64, exclude_nn=False)
        assert excl['coarse_score'].mean() < 0.05, (
            f"n_train={n_train}: near-memorization should score ~0 with exclude_nn, "
            f"got {excl['coarse_score'].mean():.3f}")
        # without exclusion the NN is drawn ~1/n_train of the time, each such draw
        # contributing ratio ~1, so the score floors near 1/n_train
        expected_floor = 1.0 / n_train
        got = incl['coarse_score'].mean().item()
        assert abs(got - expected_floor) < 0.5 * expected_floor, (
            f"n_train={n_train}: expected un-excluded score ~1/n_train={expected_floor:.3f}, "
            f"got {got:.3f}")


def test_sde_sampler_contracts_on_each_device():
    """Regression test for the MPS storage-offset bug: multiplying a device tensor by a
    CPU 0-dim view of time_steps froze the noise schedule at sigma_max, so the reverse
    SDE never contracted (samples stayed at std ~57 instead of collapsing).
    With train data = {0}, the GMM score is exactly -x/sigma^2 and the reverse SDE must
    contract samples to ~0 on every device."""
    devices = ["cpu"]
    if torch.backends.mps.is_available():
        devices.append("mps")
    ve = score_models.VE_EDM()
    for dev in devices:
        train = torch.zeros(1, 64, device=dev)
        score = score_models.GMM_score(train, ve.marginal_prob_mean, ve.marginal_prob_std)
        torch.manual_seed(0)
        latents = torch.randn(4, 64, device=dev)
        out = ve.SDEsampler(score, latents, num_steps=300)
        final_std = out.std().item()
        assert final_std < 1.0, (
            f"[{dev}] reverse SDE did not contract (final std {final_std:.2f}); "
            "time schedule is likely frozen (MPS storage-offset bug)")


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS  {fn.__name__}")
    print(f"\nAll {len(fns)} checks passed.")
