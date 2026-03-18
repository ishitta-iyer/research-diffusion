# research-diffusion

Research on multiscale diffusion models, focusing on biased vs unbiased data generation, score-based models, and memorization analysis on radial band data.
Best entry points:

src/diffusion_score_models.py — main diffusion utilities
src/multiband_data_utils.py — helpers for Matérn fields, radial band masks, Fourier filtering
src/scripts/analyze_coarse_fine_memorization.py — cleanest version of the memorization analysis
notebooks/multiscale/generate_biased_unbiased_multiband_dataset.ipynb — main experiment notebook
notes/project_map.md — full breakdown of what lives where and why

# Structure
src/          canonical Python code
notebooks/    experiment notebooks
results/      figures and saved datasets
notes/        project map and reading notes

See notes/project_map.md for a full walkthrough.
