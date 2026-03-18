# Project Map

A breakdown of what's in this repo and how it fits together.

## Canonical Code (`src/`)

- `score_models.py` — main diffusion and score model classes. Everything else imports from here, taken from 
- `radial_band_utils.py` — reusable helpers for Matérn field generation, radial band masking, Fourier filtering, and synthetic bias injection.
- `scripts/radial_band_data_generator.py` — exported from the notebook of the same name; generates multiband synthetic data and inspects spectra.
- `scripts/radial_data_mem.py` — exported from the memorization notebook; cleaner to read than the notebook version if you want the logic without the output cells.
- `tests/test_mnist_download.py` — environment check for MNIST download, not really a test suite.

## Notebooks (`notebooks/`)

- `mnist_ddpm.ipynb` — standalone MNIST DDPM experiment.
- `multiscale/` — the main experiment notebooks. This is where most of the actual work is.
- `archive/` — older or incomplete notebooks. Kept for reference.

## Results (`results/`)

- `data/` — saved dataset artifacts (e.g. the biased vs unbiased `.pt` files used across experiments).
- `figures/` — exported plots and PDFs from the experiments.

## Notes (`notes/`)

- `papers/` — background reading, kept locally and gitignored.
- `archive/` — local recovery material, also not pushed.

## Suggested Reading Order

1. `src/score_models.py` — understand the model classes first
2. `src/radial_band_utils.py` — then how the synthetic data is built
3. `src/scripts/radial_data_mem.py` — cleanest version of the memorization analysis
4. `notebooks/multiscale/multiscale_data_generator_v1.ipynb` and `radial_data_mem.ipynb` — the actual experiments with outputs
5. `results/figures/` — what came out of those runs

## What to skip

- `notes/archive/` and `notebooks/archive/` — not relevant to the current work
- `__pycache__/`, `.venv/`, `.git/` — tooling, ignore these