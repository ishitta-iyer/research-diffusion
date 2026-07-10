# Memorization in Multiscale Diffusion Models

Code for studying how score-based diffusion models memorize training data across
frequency scales. We generate synthetic Matérn Gaussian fields with controlled
frequency-band structure, optionally inject a localized bias in Fourier space,
train diffusion models (closed-form GMM scores and EDM-style U-Nets), and measure
memorization per wavenumber band — asking in particular whether coarse (low-k)
structure is memorized while fine (high-k) structure remains novel, and whether
Tikhonov-type regularization mitigates it.

## Repository structure

```
src/                          canonical Python code
  diffusion_score_models.py   VP/VE diffusion models, SDE/ODE samplers,
                              GMM / Tikhonov-regularized / empirical-Bayes score estimators
  edm.py                      EDM preconditioning wrapper and U-Net training loop
                              (with optional Tikhonov penalty)
  multiband_data_utils.py     Matérn field generation, radial bandpass masks,
                              Fourier-space bias injection
  project_paths.py            repo-relative path helpers
  scripts/                    clean script exports of the main notebook logic
notebooks/                    experiment notebooks (see guide below)
results/figures/              exported result figures
results/data/                 dataset artifacts (gitignored — regenerate locally)
notes/project_map.md          full breakdown of what lives where
```

## Experiment notebooks (`notebooks/multiscale/`)

Data generation:
- `generate_two_scale_matern_dataset.ipynb` — two-scale (coarse + fine) Matérn dataset
- `generate_biased_unbiased_multiband_dataset.ipynb` — main multiband dataset with a
  Fourier-space bias injected at a chosen wavenumber band (script export:
  `src/scripts/generate_multiband_synthetic_data.py`)
- `generate_multiband_dataset_prototype.ipynb` — earlier prototype of the generator,
  kept for reference

Training:
- `train_deepinv_on_biased_multiband_data.ipynb` — DeepInv DiffUNet on biased multiband data
- `train_vp_sde_on_biased_multiband_data.ipynb` — VP-SDE score model on biased multiband data

Memorization analysis:
- `memorization_regime_sweep.ipynb` — core experiment suite: sweeps training-set size,
  computes the per-wavenumber memorization ratio (distance to nearest training neighbour
  vs. random training sample, per radial Fourier ring) for GMM scores and U-Net
  checkpoints, and tests scale-selective memorization (coarse memorized, fine novel)
- `edm_tikhonov_memorization.ipynb` — trains EDM U-Nets with Tikhonov regularization
  (ad-hoc at sampling time vs. training-time penalty) and measures the effect on
  per-band memorization
- `analyze_multiband_memorization.ipynb` — nearest-neighbour memorization analysis on
  the multiband dataset (script export: `src/scripts/analyze_coarse_fine_memorization.py`)
- `analyze_multiband_memorization_alt_weights.ipynb` — variant of the analysis with an
  alternative band-weighting scheme
- `gaussian_field_standard_interpolation_schedule.ipynb` /
  `gaussian_field_noisy_spectral_interpolation.ipynb` — diffusion-schedule comparisons
  on Gaussian fields, with and without spectral noise

Other:
- `notebooks/mnist_ddpm.ipynb` — standalone MNIST DDPM sanity experiment

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Data

Dataset artifacts (`results/data/*.pt`) are not tracked. Regenerate them with
`notebooks/multiscale/generate_biased_unbiased_multiband_dataset.ipynb` or
`src/scripts/generate_multiband_synthetic_data.py`; downstream notebooks load them
via the paths in `src/project_paths.py`.

## Suggested reading order

1. `src/diffusion_score_models.py` — model classes and samplers
2. `src/multiband_data_utils.py` — how the synthetic data is built
3. `src/scripts/analyze_coarse_fine_memorization.py` — the memorization metric, cleanly
4. `notebooks/multiscale/memorization_regime_sweep.ipynb` — the main results
5. `notes/project_map.md` — full walkthrough
