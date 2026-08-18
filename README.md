# Scale-dependent memorization in diffusion models

This repository studies how diffusion models memorize structure at different Fourier scales.
The primary experimental line uses multiband Matérn random fields rescaled to match the
training-set geometry of the two-rectangle experiment in Baptista et al., trains the same
SongUNet/EDM family, measures nearest-neighbour memorization by frequency band, and compares
isotropic with covariance-weighted Tikhonov regularization. The current extension repeats the
successful rescaled `n_train=2` comparison across dataset size.

## Upstream provenance

The learned-model path derives from Baptista, Dasgupta, Kovachki, Oberai, and Stuart,
[*Memorization and Regularization in Generative Diffusion Models*](https://arxiv.org/abs/2501.15785),
and their public [DiffusionModelDynamics repository](https://github.com/baptistar/DiffusionModelDynamics).
Source parity is pinned to upstream commit
`2719b5d50601deb4f17fdf5306b6e26495ba19f4`. Symbol-level parity and intentional adaptations
are enforced by `src/tests/test_songunet_parity.py`.

The active Matérn experiments intentionally adapt the upstream 64×64 rectangle setup to 128×128
fields, move attention from resolution 16 to the corresponding deepest resolution 32, and use
`model_channels=16` (880,097 parameters) instead of width 128. The architecture remains the
same SongUNet class through EDMPrecond.

## Canonical experiment

1. Reproduce the upstream two-rectangle experiment.
2. Generate seed-42 four-band Matérn fields and rescale the first pair to the rectangle pair's
   Euclidean separation, `sqrt(340)`.
3. Train SongUNet through EDMPrecond with the upstream-compatible loss and training mechanics.
4. Score generated samples against training fields in the coarse `(0.5,4)`, mid1 `(4,10)`,
   mid2 `(10,18)`, and fine `(18,32)` Fourier bands.
5. Compare unregularized, isotropic, empirical-covariance, and analytic-covariance Tikhonov
   arms under a matched regularization budget.
6. Extend the completed 25-arm `n_train=2` result to nested dataset sizes
   `{2,4,8,16,32}`.

The early SongUNet notebooks are diagnostic/capacity precursors, not the headline experiment.
Their saved results show that memorization depends strongly on training-set geometry and model
capacity: unrescaled or insufficient-capacity configurations showed weak or absent
memorization. They do not support a claim that SongUNet generally fails to memorize.

## Active notebook reading order

All active notebooks live in `notebooks/multiscale/`:

1. `rectangles_n2_reproduction.ipynb` — upstream rectangle architecture/training reproduction.
2. `config_matern_n2.ipynb` and `matern_n2_reproduction.ipynb` — Matérn construction,
   rectangle-matched rescaling, and geometry comparison.
3. `songunet_memorization_vs_iteration.ipynb` and
   `songunet_perband_memorization.ipynb` — diagnostic/capacity precursors on rescaled fields.
4. `gmm_rescaled_covariance_tikhonov.ipynb` and
   `gmm_tikhonov_variants_comparison.ipynb` — exact-score controls.
5. `songunet_covariance_tikhonov.ipynb` — completed 25-arm rescaled SongUNet comparison at
   `n_train=2`.
6. `songunet_covariance_tikhonov_dataset_size.ipynb` — active nested dataset-size extension.

## Repository layout

```text
src/
  songunet.py                 canonical SongUNet, EDMPrecond, and active factory
  songunet_edm.py             upstream EDM loss/sampler/training mechanics and active penalties
  multiband_data_utils.py     Matérn generation, rescaling, and covariance-spectrum helpers
  memorization_metrics.py     locked per-ring and per-band memorization metric
  diffusion_score_models.py   closed-form GMM scores and VP/VE diffusion utilities
  edm.py                      historical SmallUNet-oriented EDM path (preserved)
  unet.py                     historical SmallUNet architecture (preserved)
  tests/                      consolidation and SongUNet parity/regression tests
notebooks/
  multiscale/                 active experiments and retained non-SmallUNet studies
  archived/smallunet/         byte-identical historical SmallUNet notebooks
results/
  figures/                    active and retained non-SmallUNet figures
  data/                       ignored active `.pt` artifacts
  archived/                   local-only historical results (ignored by Git)
```

## Reproducibility conventions

- Active learned model: SongUNet through EDMPrecond, `[2,2,2]` channel multipliers, attention
  at 32 for 128×128 fields, `model_channels=16`, `sigma_data=0.5`.
- Active sampler: deterministic 40-step EDM/Heun, `sigma_min=0.002`, `sigma_max=80`, `rho=7`,
  no churn. The archived SmallUNet lineage uses a different 10/1,000 convention.
- Geometry: normalized seed-42 Matérn pool; one fixed scalar matches the first pair to
  `sqrt(340)`. Dataset-size arms are nested and preserve the exact completed first pair.
- Metric: `exclude_nn=True`, `mean_of_ratios`, generated-sample ring normalization, and the
  four fixed bands above. On an `n_train` axis, report against the per-`n_train` neutral
  held-out baseline.
- Fixed seeds: dataset 42; training and latent/reference seeds are recorded by each notebook
  and saved artifact.

Run the regression suites on CPU with:

```bash
python3 src/tests/test_consolidation.py
python3 src/tests/test_songunet_parity.py
```

## Data and result policy

Model weights, checkpoints, and `.pt` artifacts are ignored by Git. Active local outputs live
under `results/data/`, including `baptista_rectangles_n2/`, `baptista_matern_n2/`,
`songunet_cov_tikhonov/`, and `songunet_cov_tikhonov_dataset_size_full/`. Cluster runs write to
the same repository-relative structure or an explicitly configured result directory; copy their
completed artifacts back without force-adding them.

Historical SmallUNet notebooks are preserved under `notebooks/archived/smallunet/`. All
historical result files under `results/archived/` are local-only and ignored by Git. Historical
documentation remains available locally, including producer and validity records and the unsafe
`DO NOT CITE` artifact warning.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
