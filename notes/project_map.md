# Project Map

A breakdown of what's in this repo and how it fits together.

Reference throughout: Baptista, Dasgupta, Kovachki, Oberai, Stuart, *Memorization and
Regularization in Generative Diffusion Models*, [arXiv:2501.15785](https://arxiv.org/abs/2501.15785).

## Canonical Code (`src/`)

- `diffusion_score_models.py` — diffusion processes (`VP`, `VE`, `VE_EDM`), SDE/ODE samplers,
  and the closed-form score classes:
  - `GMM_score` — the empirical-score minimizer, `s = (m(t)·x̄ − x)/σ²` (paper Thm 3.2; this is
    the memorizing solution).
  - `GMM_score_TikhonovRegularized` — isotropic Tikhonov, denominator `σ² + c` (paper eq. 5.7,
    `Γ(t) = (c/σ²)I`).
  - `GMM_score_CovarianceTikhonov` — **covariance-weighted** Tikhonov, per-mode denominator
    `σ² + c/λ(k)` (paper eq. 5.3 with matrix `Γ(t) = (c/σ²)Σ⁻¹`). `weighting='direct'` gives the
    anti-weighted control `σ² + c·λ(k)`. Null modes left unregularized (pseudo-inverse).
  - `GMM_score_EmpiricalBayes` — the paper's §5.2 alternative regularizer.
- `edm.py` — EDM (Karras et al. 2022) preconditioning, training, sampling.
  - `train_edm(...)` — denoising score matching. Two **default-off** regularizers:
    `c_tikhonov` (isotropic) and `c_tikhonov_cov` + `cov_spectrum` (covariance-weighted).
  - `tikhonov_penalty(...)` — the shared penalty term (used by both the trainer and its tests).
  - `EDMScoreWrapper` — denoiser → score bridge, with a *sampling-time* `c_tikhonov` knob.
    Note: training-time and sampling-time Tikhonov implement the **same** `σ² → σ² + c` shift;
    use one or the other, never both.
- `unet.py` — `SmallUNet(base_channels, emb_dim, num_levels)`. `num_levels` is the depth knob;
  the default `(16, 64, 3)` reproduces the old inline definition exactly (~195k params,
  verified layer-for-layer by tests). `remap_legacy_state_dict` loads pre-consolidation weights.
- `memorization_metrics.py` — the per-wavenumber memorization metric.
  `RingMetricContext(N, bands).evaluate(x_gen, x_train, exclude_nn=...)` returns
  `ratio(k) = err_to_NN / err_to_random` (<1 ⇒ memorized) plus per-band scores.
  **Use `exclude_nn=True` whenever comparing across training-set sizes** — otherwise the NN is
  part of its own baseline and the score floors at ~1/n_train (0.5 at n_train=2!).
- `device_utils.py` — `resolve_device()`: cuda > mps > cpu. New notebooks call this once at the
  top; pass `resolve_device("cpu")` to force CPU on another machine.
- `multiband_data_utils.py` — Matérn field generation, radial band masks, Fourier filtering,
  synthetic bias injection.
- `scripts/` — heavy standalone jobs (`denoiser_gap.py`, `basin_reconstruction.py`,
  `coordconv_experiment.py`) and clean notebook exports.
- `tests/test_consolidation.py` — run `python3 src/tests/test_consolidation.py`. Pins the score
  reductions, the U-Net equivalence, the two Tikhonov **stationary points** against the paper's
  closed forms, and the MPS sampler regression (below). On the metric it pins only that
  `exclude_nn=True` removes the 1/n_train floor — i.e. near-total memorization scores ≈ 0 at
  n_train ∈ {2,4,8}, the regime where the ratio is pinned near zero. It does **not** pin
  n_train-independence in the interesting regime, and the metric is not n_train-independent
  (see Gotchas).

## Notebooks (`notebooks/`)

- `mnist_ddpm.ipynb` — standalone MNIST DDPM experiment.
- `multiscale/` — the main experiments:
  - **Data generation:** `generate_two_scale_matern_dataset.ipynb`,
    `generate_biased_unbiased_multiband_dataset.ipynb`, `generate_multiband_dataset_prototype.ipynb`.
  - **Training on biased data:** `train_deepinv_on_biased_multiband_data.ipynb`,
    `train_vp_sde_on_biased_multiband_data.ipynb`.
  - **Core memorization analysis:** `memorization_regime_sweep.ipynb` (per-wavenumber ratio vs
    training-set size, GMM vs U-Net, scale-selective memorization);
    `analyze_multiband_memorization.ipynb` (+ `_alt_weights`).
  - **Regularization:** `edm_tikhonov_memorization.ipynb` (isotropic, sampling- vs training-time);
    `gmm_covariance_tikhonov_memorization.ipynb` (first covariance-weighted study);
    `gmm_tikhonov_variants_comparison.ipynb` (**the controlled comparison** — see below);
    `edm_unet_covariance_tikhonov.ipynb` (training-time covariance penalty in the U-Net).
  - **U-Net memorization mechanism:** `edm_unet_train_size_sweep.ipynb`,
    `edm_unet_train_size_sweep_sigma_fix.ipynb` (sampler ablation: σ-range and discretization
    both null → σ_max = 10 kept for consistency, see Gotchas),
    `edm_unet_memorization_mechanism.ipynb` (denoiser gap, basin reachability,
    CoordConv: memorization is *in* the net but unreachable from pure noise).
  - **Transition / capacity:** `edm_unet_memorization_transition.ipynb` (n_train × training time),
    `edm_unet_capacity_sweep.ipynb` (width × depth).
  - **Schedules:** `gaussian_field_standard_interpolation_schedule.ipynb`,
    `gaussian_field_noisy_spectral_interpolation.ipynb`.

## Results (`results/`)

- `data/` — saved `.pt` artifacts (gitignored; regenerate by running the notebooks, all seeded).
- `figures/` — exported plots.

## Gotchas

- **MPS sampler bug (fixed).** `DiffusionModel.SDEsampler` built `time_steps` on the CPU; on MPS,
  multiplying a device tensor by a CPU 0-dim *view* silently reads storage offset 0, freezing the
  noise schedule at σ_max so the reverse SDE never contracted. Symptom: generated fields with
  std ≈ 57 and every memorization score ≈ 1 (looked like "regularization works", was a bug).
  `time_steps` is now created on the sampler's device; `test_sde_sampler_contracts_on_each_device`
  guards it. **Any MPS result produced before this fix is invalid.**
- **Metric floor at small n_train.** See `exclude_nn` above.
- **Metric convention, and why a coarse score below 1 is not evidence on its own.** Two
  conventions are locked, and as of 2026-08-04 `paper/main.tex` is being changed to state them
  (the code is correct and unchanged):
  - *Aggregation*: **mean-of-ratios**, `mean_j(e_NN / e_rand_j)` — the `evaluate` default, used
    by every committed number. `ratio_of_means` remains available as an alternative
    aggregation (effect: −0.031 coarse).
  - *Normalization*: `ring_rel_l2` divides **both** terms by the **generated sample's** ring
    norm, so the normalization cancels and the metric is a pure error ratio (normalizing by the
    reference field instead: +0.073 coarse).

  Neither convention makes the score n_train-independent. Two confounds survive: (i) a Jensen
  gap of ~0.03 coarse under mean-of-ratios, exactly 0 at n_train=2 where only one non-NN
  reference exists so `err_rand` has zero variance and the two aggregations coincide *exactly*;
  (ii) the larger one — the NN is an **argmin over n_train candidates**, so the best-of-n match
  improves and the neutral score falls as n_train grows. Held-out real fields (perfect
  generalization), `exclude_nn=True`, coarse band, mean-of-ratios:
  0.8567 (n=2), 0.9050 (4), 0.8448 (8), 0.8150 (16), 0.8126 (32).
  **Rule:** on an n_train axis always report scores against the per-n_train neutral baseline
  computed under the same convention. Switching aggregation is not the fix.
- **σ_max = 10 is a consistency choice, not a fix.** `edm_unet_train_size_sweep_sigma_fix.ipynb`
  pre-registered a decision rule over four sampler configs — A (σ_max 80, 500 steps), B (80, 2000),
  C (10, 500), D (10, 1000): *C ≈ D ≪ A ≈ B* ⇒ σ-range is the cause; *B ≪ A, C ≈ A* ⇒
  discretization; *all ≈ A* ⇒ neither explains it. The measured coarse ratios are
  **all ≈ A** — n=4: 0.9112 / 0.8954 / 0.8773 / 0.8959; n=32: 0.8081 / 0.8322 / 0.8111 / 0.8028,
  a spread of 0.005–0.034, inside the 0.03–0.06 seed noise. So the ablation is **null**: the
  sampler was ruled out, not fixed. σ_max = 10 is retained everywhere purely so all U-Net runs
  are mutually comparable. Do **not** describe this as a diagnosed train/sample σ mismatch.
  The ablation's own third branch points at the *training* σ distribution (broader `P_mean`/
  `P_std`) — still open, and the same axis as the σ_max-vs-spacing question.
- **`length_scale`** in `generate_matern_laplace` is the spectral shift ℓ² in `(|k|²+ℓ²)^{-s}`, so
  *larger* values mean *finer* fields. The hard band masks make the actual band unambiguous.
- **Hermitian symmetry.** A power spectrum of real data satisfies λ(k) = λ(−k). Hand-supplied
  spectra are symmetrized defensively (`hermitian_symmetrize`); an asymmetric weighting has no
  real-valued minimizer and the per-mode closed form would silently not hold.

---

# Current Status & Next Steps

_Last updated: 2026-07-25._

## Where things stand

**Audit of the recent commits: the math is sound and matches the paper.** Verified first-hand:
`GMM_score_TikhonovRegularized` and `EDMScoreWrapper` implement eq. (5.7); the `train_edm`
isotropic penalty has stationary point `s* = s_true/(1+c/σ²)` (the EDM loss weight cancels);
the covariance-weighted class is the paper's matrix-`Γ` case with `Γ = (c/σ²)Σ⁻¹`, which the
paper permits but never tests. The worry about breaking source files was unfounded — commit
`12fe131` was purely additive, and the only shared-source change (`c_tikhonov` in `train_edm`,
`8fff7c6`) is gated behind a default that preserves the old loss exactly.

**Result 1 — covariance weighting is scale-selective** (`gmm_tikhonov_variants_comparison.ipynb`,
n_train = 32, closed-form GMM). Band scores (coarse / fine; <1 = memorized):

| c | isotropic | budget-matched iso | covariance | anti-weighted |
|---|---|---|---|---|
| 0 | 0.037 / 0.045 | 0.037 / 0.045 | 0.037 / 0.045 | 0.037 / 0.045 |
| 1e-3 | 0.038 / 0.165 | 0.044 / 0.603 | 0.037 / **0.632** | 0.061 / 0.060 |
| 1e-2 | 0.041 / 0.409 | 0.060 / 0.917 | 0.037 / **0.919** | 0.110 / 0.105 |
| 1 | 0.078 / 0.972 | 0.245 / 0.999 | **0.040** / 0.999 | 0.593 / 0.589 |

Reading: covariance weighting de-memorizes the **fine** band ~100× earlier in `c` than plain
Tikhonov while the **coarse** band stays memorized (0.037 → 0.043) at every `c` tested. The
budget-matched control is the sharpest comparison: it matches the covariance variant's fine-band
curve almost exactly, but pays for it in the coarse band (0.037 → 0.44 at c = 5, where covariance
stays at 0.043). So the same *amount* of regularization spread uniformly buys the same fine-scale
de-memorization while destroying coarse-scale structure — the selectivity comes from the
**allocation across modes, not the magnitude**. This is the direct answer to Prof. Baptista's
question. The anti-weighted control (`c·λ(k)`) de-memorizes
coarse first, the mirror image, showing direction-specificity. The population-spectrum variant
is indistinguishable from the empirical one (per-mode estimation error ~16%).
Figures: `results/figures/tikvariants_{per_k,band_scores,generated_spectra}.png`.

**Result 2 — U-Net memorization mechanism** (earlier work, still standing): the EDM U-Net
barely memorizes from pure noise (coarse ratio 0.81–0.91) even though memorization is present
in the weights (train/held-out denoiser gap up to ~15× at σ ≲ 2; memorized basins 100%
reachable from `y + σ₀ε`). Diagnosis: the U-Net is an *amplifier* of whatever coarse content is
in its initialization, whereas the GMM is a *classifier* of it. The sampler was **ruled out, not
fixed** — see Gotchas; σ_max = 10 is a consistency choice, not a correction.

## Next steps

Ordered; each notebook has a `SMOKE = True` switch for a minutes-long end-to-end check, and
saves checkpoints incrementally so evaluation can be rerun without retraining.

1. **Run `edm_unet_memorization_transition.ipynb`** (Prof. Baptista ask #2: show the
   memorization ↔ generalization transition for the U-Net). n_train ∈ {2,4,8,16,32} ×
   30k steps with log-spaced checkpoints, matched sampler (σ_max = 10, 1000 steps).
   Reports the per-band ratio *and* the Baptista-style pixel-space collapse fraction, with the
   GMM closed form as the memorization ceiling. ~1.5–2.5 h.
   *If nothing memorizes by 30k steps, extend the small-n_train runs to ~50k (the paper's N=2
   image U-Net needed that) before concluding the EDM U-Net resists memorization.*

2. **Run `edm_unet_capacity_sweep.ipynb`** (ask #3: "does it change with layers / parameters?").
   Width `base_channels ∈ {8,16,32,64}` at 3 levels and depth `num_levels ∈ {2,3,4}` at C=16, at
   fixed n_train = 8 — 59k → 2.7M params. Paper prediction (Fig. 18): all sizes eventually
   memorize, larger ones faster. A small net plateauing above the ceiling would be genuine
   architecture-induced regularization and worth reporting on its own.

3. **Run `edm_unet_covariance_tikhonov.ipynb`** (the synthesis). Trains with the covariance
   penalty and checks whether the U-Net inherits the closed form's scale-selectivity. **Depends
   on step 1**: pick a training length where the *unregularized* baseline actually memorizes —
   regularization can only be shown to remove memorization that was there to begin with.

4. **Write up for the reply email / paper.** The Tikhonov-variants table plus the transition and
   capacity curves answer all three of Prof. Baptista's asks. Worth flagging to him: the
   covariance weighting is a matrix-`Γ` instance his framework allows but the paper does not
   explore, and the per-band metric is what makes the selectivity visible at all.

## Open questions to raise

- **Null modes.** Σ is singular (band-limited data ⇒ λ = 0 for k > 32 and DC). We use
  pseudo-inverse semantics: those modes get no regularization, so the unregularized score keeps
  pulling them to zero. A ridge (Σ + δI) would instead give them effective constant c/δ and the
  reverse SDE would fill them with noise. Worth confirming this is the intended choice.
- **Empirical vs population Σ.** Currently the empirical spectrum from the training set; the
  population spectrum is known analytically here. Shown not to matter qualitatively.
- Whether to report the pixel-space collapse fraction (comparable to the paper's Figs. 5/9/11/14)
  or the per-band ratio as the primary metric — the per-band one is this project's contribution.
