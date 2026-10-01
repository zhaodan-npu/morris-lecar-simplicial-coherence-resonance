#!/bin/bash
#SBATCH --job-name=ml-cr-main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=24:00:00
#SBATCH --array=0-29
#SBATCH --output=logs/main_%A_%a.out
#SBATCH --error=logs/main_%A_%a.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export MPLBACKEND=Agg
export PYTHONDONTWRITEBYTECODE=1
PYTHON_BIN="${PYTHON_BIN:-python}"
"$PYTHON_BIN" reproduce.py run-network --family main \
  --experiment candidate39_main --network "${SLURM_ARRAY_TASK_ID:?}" \
  --output "${ML_OUTPUT_ROOT:-results}"
