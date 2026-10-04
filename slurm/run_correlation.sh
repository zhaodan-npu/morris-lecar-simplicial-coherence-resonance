#!/bin/bash
#SBATCH --job-name=ml_correlation
#SBATCH --output=logs/correlation_%A_%a.out
#SBATCH --error=logs/correlation_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=12G
#SBATCH --time=04:00:00
#SBATCH --array=0-599%40

set -euo pipefail

CODE_DIR="${SLURM_SUBMIT_DIR:?Submit from the repository root}"
RUN_ROOT="${RUN_ROOT:-$CODE_DIR/results/correlation}"
PYTHON_BIN="${PYTHON_BIN:-python}"
ALPHA="${ALPHA:?Set ALPHA to 1.5 or 2.0}"
case "$ALPHA" in
    1.5|2.0) ;;
    *) echo 'ALPHA must be 1.5 or 2.0' >&2; exit 2 ;;
esac

NETWORK_INDEX=$((SLURM_ARRAY_TASK_ID / 20))
DV_INDEX=$((SLURM_ARRAY_TASK_ID % 20))
TASK_DIR="$RUN_ROOT/batches/a${ALPHA}_n${NETWORK_INDEX}_d${DV_INDEX}"
mkdir -p "$TASK_DIR" "$RUN_ROOT/results"
cd "$CODE_DIR"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1 MPLBACKEND=Agg
export CODE_DIR RUN_ROOT PYTHON_BIN ALPHA NETWORK_INDEX DV_INDEX TASK_DIR

run_batch() {
    local trial="$1"
    local label
    for label in 0 1; do
        "$PYTHON_BIN" "$CODE_DIR/morris_lecar_voltage_coherence_true.py" condition \
            --output-dir "$TASK_DIR" --alpha "$ALPHA" \
            --network-index "$NETWORK_INDEX" --dv-index "$DV_INDEX" \
            --trial-index "$trial" --label-index "$label" \
            --n-paths 12 --sampled-nodes 12 --trace-stride 10 \
            --max-lag-ms 50000 --t-total 120000 --t-warmup 3000
    done
}
export -f run_batch
seq 0 11 | xargs -n 1 -P "$SLURM_CPUS_PER_TASK" bash -c 'run_batch "$1"' _

"$PYTHON_BIN" "$CODE_DIR/morris_lecar_voltage_coherence_true.py" combine \
    --input-dir "$TASK_DIR" --output-dir "$RUN_ROOT/results" \
    --alpha "$ALPHA" --network-index "$NETWORK_INDEX" --dv-index "$DV_INDEX" \
    --n-trial-batches 12 --n-paths 12
