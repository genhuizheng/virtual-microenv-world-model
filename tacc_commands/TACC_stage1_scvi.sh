#!/bin/bash
#SBATCH -J vm_scvi_s1
#SBATCH -o vm_scvi_s1_%j.out
#SBATCH -e vm_scvi_s1_%j.err
#SBATCH -p gh
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --mail-user=genhuizheng@utexas.edu
#SBATCH --mail-type=all
#SBATCH -t 12:00:00
#SBATCH -A MCB26031

# Usage:
#   bash   tacc_commands/TACC_stage1_scvi.sh test
#   sbatch tacc_commands/TACC_stage1_scvi.sh full paired_training_h5ad_50k 100 20260824_scvi 0
#   sbatch tacc_commands/TACC_stage1_scvi.sh full paired_training_h5ad_50k 100 20260824_scvi 1
#   ... (repeat fold-index 0..4 for 5-fold coverage; each is a separate sbatch submission)
# Arguments: mode, data-dir, max-epochs, run tag, fold-index.

source /home1/10119/ghzheng/.bashrc
conda activate worldmodeltraining

# Locate the repo root without a hardcoded path. Under `sbatch`, SLURM copies
# this script into a per-job spool directory before running it, so $0 no
# longer points at its real location -- SLURM_SUBMIT_DIR (the directory
# `sbatch` was invoked from) is the reliable way to get back to the repo
# root there. Direct `bash tacc_commands/TACC_stage1_scvi.sh ...` runs don't
# set SLURM_SUBMIT_DIR, so fall back to $0's own location in that case.
cd "${SLURM_SUBMIT_DIR:-$(dirname "$(readlink -f "$0")")}"
set -e

MODE="${1:-full}"
DATA_DIR="${2:-paired_training_h5ad_50k}"
RUN_TAG="${4:-$(date +%Y%m%d)}"
FOLD_INDEX="${5:-0}"
NUM_FOLDS=5

case "$MODE" in
  test)
    MAX_EPOCHS=2
    EXTRA_ARGS=(--max-cells 2000)
    RUN_TAG="${RUN_TAG}_test"
    ;;
  full)
    MAX_EPOCHS="${3:-100}"
    EXTRA_ARGS=()
    ;;
  *)
    echo "ERROR: mode must be test or full"
    exit 2
    ;;
esac

test -f train_stage1_scvi.py || {
  echo "ERROR: missing train_stage1_scvi.py"
  exit 1
}

test -d "$DATA_DIR" || {
  echo "ERROR: missing data dir: $DATA_DIR"
  exit 1
}

VAE_DIR="stage1_${RUN_TAG}_scvi_$(basename "$DATA_DIR")"
VAE_CHECKPOINT="$VAE_DIR/best_stage1_scvi_fold${FOLD_INDEX}.pt"

echo "============================================================"
echo "Stage 1 only: scVI (negative-binomial VAE)"
echo "python=$(command -v python)"
echo "mode=$MODE"
echo "data_dir=$DATA_DIR"
echo "run_tag=$RUN_TAG"
echo "maximum_epochs=$MAX_EPOCHS"
echo "fold=${FOLD_INDEX}/${NUM_FOLDS}"
echo "output=$VAE_DIR"
echo "============================================================"

python train_stage1_scvi.py \
  --data-dir "$DATA_DIR" \
  --checkpoint-dir "$VAE_DIR" \
  --max-epochs "$MAX_EPOCHS" \
  --batch-size 256 \
  --lr 1e-3 \
  --latent-dim 128 \
  --n-hidden 128 \
  --n-layers 1 \
  --dispersion gene \
  --gene-likelihood zinb \
  --num-folds "$NUM_FOLDS" \
  --fold-index "$FOLD_INDEX" \
  --group-column patient_id \
  "${EXTRA_ARGS[@]}" \
  --device cuda \
  --overwrite

test -f "$VAE_CHECKPOINT" || {
  echo "ERROR: scVI checkpoint was not created: $VAE_CHECKPOINT"
  exit 1
}

echo "============================================================"
echo "Stage 1 (scVI) finished"
echo "best_checkpoint=$VAE_CHECKPOINT"
echo "Review $VAE_DIR/stage1_scvi_fold${FOLD_INDEX}_history.csv before Stage 2."
echo "Stage 2 example:"
echo "  python train_five_level_world_model.py --data-dir $DATA_DIR --level 2 \\"
echo "    --stage1-scvi-checkpoint $VAE_CHECKPOINT --expression-transform none \\"
echo "    --group-column patient_id --num-folds $NUM_FOLDS --fold-index $FOLD_INDEX --latent-dim 128"
echo "============================================================"
