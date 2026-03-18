# Project Map

This file explains what each part of the repository is for so the project is readable without reverse-engineering filenames.

## Canonical Code

- `src/score_models.py`: main diffusion and score-model utilities. This should be treated as the canonical copy.
- `src/radial_band_utils.py`: reusable helpers for Matérn field generation, radial band masks, Fourier filtering, and synthetic bias injection.
- `src/scripts/radial_band_data_generator.py`: notebook-exported script for generating multiband synthetic data and inspecting spectra.
- `src/scripts/radial_data_mem.py`: notebook-exported script for the memorization-style analysis on biased versus unbiased multiscale data.
- `src/tests/test_mnist_download.py`: small environment check for downloading MNIST into the repository's results folder.

## Notebooks

- `notebooks/mnist_ddpm.ipynb`: separate MNIST DDPM experiment.
- `notebooks/multiscale/`: main multiscale diffusion notebooks. These are the most relevant experiment records for the professor to inspect.
- `notebooks/archive/`: placeholder or incomplete notebooks that should not be treated as the main story of the project.

## Results

- `results/data/multiscale_unbiased_vs_biased.pt`: saved dataset artifact used by later experiments.
- `results/figures/`: exported figures and PDFs generated during experimentation.

## Notes

- `notes/papers/`: background reading PDFs kept locally and ignored from Git.
- `notes/archive/`: local-only archive material kept for recovery, not for GitHub.

## Suggested Reading Order

1. Read `src/score_models.py` to understand the diffusion model classes.
2. Read `src/radial_band_utils.py` to understand how the synthetic multiscale data is generated.
3. Inspect `src/scripts/radial_data_mem.py` for the clearest Python version of the memorization analysis.
4. Use `notebooks/multiscale/multiscale_data_generator_v1.ipynb` and `notebooks/multiscale/radial_data_mem.ipynb` as the main experiment notebooks.
5. Look at `results/figures/` and `results/data/` for outputs produced by those experiments.

## What To Ignore First

- `notes/archive/`: local recovery material only. Do not push it.
- `notebooks/archive/`: incomplete or placeholder notebooks.
- hidden folders like `.venv/`, `__pycache__/`, and `.git/`: environment and tooling only.
