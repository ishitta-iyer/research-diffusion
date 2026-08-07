#!/bin/bash
# Baptista et al. section 5.4 rectangles, N=2 -- Figure 18 reproduction on Killarney.
#
# Runs notebooks/multiscale/baptista_rectangles_n2_reproduction.ipynb.
# Safe to resubmit: each arm resumes from its own state dump, completed arms are skipped.
#
# SUBMIT FROM THE REPO ROOT:
#   cd ~/projects/aip-baptista/ishiyer/research-diffusion
#   sbatch slurm/rectangles_n2.sh                  # all six arms, one GPU, serial
#   RECT_ARMS=128 sbatch slurm/rectangles_n2.sh    # just the 55.7M arm
#
# BEFORE FIRST USE: check the two CHECK-ME lines below against your own job_sigma.sh /
# job_multiseed.sh, which are already known to work on this cluster. Their module loads and
# venv path beat anything guessed from documentation.

#SBATCH --job-name=rect-n2
#SBATCH --account=aip-baptista
#SBATCH --gpus-per-node=l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/rect-n2-%j.out

set -euo pipefail

echo "job $SLURM_JOB_ID on $(hostname), started $(date)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

# CHECK-ME 1: match whatever job_sigma.sh already does here.
module load python/3.11
source "$HOME/envs/diffusion/bin/activate"

cd "$SLURM_SUBMIT_DIR"

# CHECK-ME 2: heavy artifacts go to scratch, NOT the shared aip-baptista project quota.
# The resume state is ~0.9 GB for the 55.7M arm, ~1.3 GB across all six. Project space is
# shared with everyone else in the allocation; scratch is yours and is not backed up.
export RECT_RESULTS_DIR="${RECT_RESULTS_DIR:-$HOME/scratch/rectangles_n2/data}"
export RECT_FIG_DIR="${RECT_FIG_DIR:-$HOME/scratch/rectangles_n2/figures}"
mkdir -p "$RECT_RESULTS_DIR" "$RECT_FIG_DIR" logs results/notebook_runs

export MPLBACKEND=Agg          # headless plotting
export PYTHONUNBUFFERED=1      # log updates live instead of at exit
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

echo "results -> $RECT_RESULTS_DIR"
python3 -c "import torch; print('torch', torch.__version__, '| cuda', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO GPU')"

# --output (not --inplace) so a killed job cannot leave the tracked notebook half-written.
# The .pt artifacts under $RECT_RESULTS_DIR are what actually carry the results.
python3 -m nbconvert --to notebook --execute \
    --ExecutePreprocessor.timeout=-1 \
    --ExecutePreprocessor.kernel_name=python3 \
    --output-dir results/notebook_runs \
    --output "rect_n2_${SLURM_JOB_ID}.ipynb" \
    notebooks/multiscale/baptista_rectangles_n2_reproduction.ipynb

echo "finished $(date)"

# ---------------------------------------------------------------------------------------------
# ARRAY VARIANT -- one arm per GPU; wall time becomes the slowest arm (~1.3 h) rather than the
# sum (~4.5 h). Replace the --time/--output lines above with:
#
#   #SBATCH --time=04:00:00
#   #SBATCH --array=0-5
#   #SBATCH --output=logs/rect-n2-%A_%a.out
#
# and insert before the nbconvert call:
#
#   ARMS=(4 8 16 32 64 128)
#   export RECT_ARMS=${ARMS[$SLURM_ARRAY_TASK_ID]}
#
# Result files are per-arm so tasks cannot clobber each other. The only shared writes are the
# three .png figures, which each task regenerates from whatever arms exist at the time --
# harmless, and you should re-run the plotting cells once at the end for the full six-curve
# figure anyway.
# ---------------------------------------------------------------------------------------------
