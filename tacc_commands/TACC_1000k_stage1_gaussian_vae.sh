#!/bin/bash
#SBATCH -J vm_1m_vae
#SBATCH -o vm_1m_vae_%j.out
#SBATCH -e vm_1m_vae_%j.err
#SBATCH -p gh
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --mail-user=genhuizheng@utexas.edu
#SBATCH --mail-type=all
#SBATCH -t 48:00:00
#SBATCH -A MCB26031

# Usage:
#   bash  tacc_commands/TACC_1000k_stage1_gaussian_vae.sh test
#   sbatch tacc_commands/TACC_1000k_stage1_gaussian_vae.sh full 50 20260805_final
# Arguments: mode, maximum epochs, shared run tag.

source /home1/10119/ghzheng/.bashrc
conda activate worldmodeltraining

cd /scratch/10119/ghzheng/Virtual_microenv
set -e

MODE="${1:-full}"
RUN_TAG="${3:-$(date +%Y%m%d)}"
case "$MODE" in
  test)
    VAE_EPOCHS=1
    LIMIT_ARGS=(--max-train-steps 5 --max-val-steps 2)
    RUN_TAG="${RUN_TAG}_test"
    ;;
  full)
    VAE_EPOCHS="${2:-50}"
    LIMIT_ARGS=(--max-val-steps 100)
    ;;
  *)
    echo "ERROR: mode must be test or full"
    exit 2
    ;;
esac

test -f train_stage1_gaussian_vae.py || {
  echo "ERROR: missing train_stage1_gaussian_vae.py"
  exit 1
}

test -d paired_training_h5ad_1000k_fast || {
  echo "ERROR: missing paired_training_h5ad_1000k_fast"
  exit 1
}

VAE_DIR="stage1_${RUN_TAG}_gaussian_vae_1000k"
VAE_CHECKPOINT="$VAE_DIR/best_stage1_vae_fold0.pt"

echo "============================================================"
echo "Stage 1 only: Gaussian VAE"
echo "python=$(command -v python)"
echo "mode=$MODE"
echo "run_tag=$RUN_TAG"
echo "maximum_epochs=$VAE_EPOCHS"
echo "output=$VAE_DIR"
echo "============================================================"

python train_stage1_gaussian_vae.py \
  --data-dir paired_training_h5ad_1000k_fast \
  --checkpoint-dir "$VAE_DIR" \
  --epochs "$VAE_EPOCHS" \
  --batch-size 4096 \
  --lr 1e-3 \
  --kl-weight 1e-3 \
  --latent-dim 128 \
  --hidden-dim 512 \
  --depth 3 \
  --num-folds 1 \
  --fold-index 0 \
  "${LIMIT_ARGS[@]}" \
  --device cuda \
  --overwrite

test -f "$VAE_CHECKPOINT" || {
  echo "ERROR: VAE checkpoint was not created: $VAE_CHECKPOINT"
  exit 1
}

echo "============================================================"
echo "Stage 1 finished"
echo "best_checkpoint=$VAE_CHECKPOINT"
echo "Review $VAE_DIR/stage1_vae_fold0_history.csv before Stage 2."
echo "============================================================"
