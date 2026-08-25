#!/bin/bash
#SBATCH -J vm_scvi_s2
#SBATCH -o vm_scvi_s2_%j.out
#SBATCH -e vm_scvi_s2_%j.err
#SBATCH -p gh
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --mail-user=genhuizheng@utexas.edu
#SBATCH --mail-type=all
#SBATCH -t 04:00:00
#SBATCH -A MCB26031

# Stage 2: train one transition-model architecture on top of a frozen,
# fold-matched Stage-1 scVI checkpoint. One (level, variant, fold) per job.
#
# Usage:
#   sbatch tacc_commands/TACC_stage2_scvi_screen.sh <level> <variant> <fold_index> [run_tag]
# Examples:
#   sbatch tacc_commands/TACC_stage2_scvi_screen.sh 1 default 0
#   sbatch tacc_commands/TACC_stage2_scvi_screen.sh 5 hierarchical_dit 3
#   sbatch tacc_commands/TACC_stage2_scvi_screen.sh 0 identity 0        # baseline
#
# Fixed for this round: 50k data, the scVI Stage-1 checkpoints already
# trained in stage1_scvi50k_scvi_paired_training_h5ad_50k/.

source /home1/10119/ghzheng/.bashrc
conda activate worldmodeltraining

cd "${SLURM_SUBMIT_DIR:-$(dirname "$(readlink -f "$0")")/..}"
set -e

LEVEL="${1:?level required (0-6; 0 is the identity baseline)}"
VARIANT="${2:-default}"
FOLD_INDEX="${3:?fold_index required}"
RUN_TAG="${4:-scvi50k_screen}"

DATA_DIR="paired_training_h5ad_50k"
STAGE1_DIR="stage1_scvi50k_scvi_paired_training_h5ad_50k"
STAGE1_CKPT="${STAGE1_DIR}/best_stage1_scvi_fold${FOLD_INDEX}.pt"

test -f "$STAGE1_CKPT" || {
  echo "ERROR: missing Stage-1 checkpoint: $STAGE1_CKPT"
  exit 1
}

VARIANT_PART=""
if [ "$VARIANT" != "default" ]; then
  VARIANT_PART="_${VARIANT}"
fi
CHECKPOINT_DIR="checkpoints_${RUN_TAG}_level${LEVEL}${VARIANT_PART}_fold${FOLD_INDEX}"

echo "============================================================"
echo "Stage 2 screen: level=$LEVEL variant=$VARIANT fold=$FOLD_INDEX"
echo "stage1_checkpoint=$STAGE1_CKPT"
echo "checkpoint_dir=$CHECKPOINT_DIR"
echo "============================================================"

# Level 0 is the non-learned identity baseline (predicts no change). It has no
# trainable parameters, so the training script runs it as a single evaluation
# pass -- but through the exact same loss/metric/checkpoint pipeline as every
# other candidate, which is what makes it a comparable sanity floor.
EPOCHS=50
if [ "$LEVEL" = "0" ]; then
  EPOCHS=1
fi

python train_five_level_world_model.py \
  --data-dir "$DATA_DIR" \
  --level "$LEVEL" \
  --variant "$VARIANT" \
  --stage1-scvi-checkpoint "$STAGE1_CKPT" \
  --expression-transform none \
  --group-column patient_id --num-folds 5 --fold-index "$FOLD_INDEX" \
  --latent-dim 128 --batch-size 256 --epochs "$EPOCHS" \
  --checkpoint-dir "$CHECKPOINT_DIR" \
  --device cuda

echo "============================================================"
echo "Stage 2 screen finished: level=$LEVEL variant=$VARIANT fold=$FOLD_INDEX"
echo "============================================================"
