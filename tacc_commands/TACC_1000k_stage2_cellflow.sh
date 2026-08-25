#!/bin/bash
#SBATCH -J vm_1m_cellflow
#SBATCH -o vm_1m_cellflow_%j.out
#SBATCH -e vm_1m_cellflow_%j.err
#SBATCH -p gh
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --mail-user=genhuizheng@utexas.edu
#SBATCH --mail-type=all
#SBATCH -t 48:00:00
#SBATCH -A MCB26031

# Usage after Stage 1 succeeds:
#   bash  tacc_commands/TACC_1000k_stage2_cellflow.sh test
#   sbatch tacc_commands/TACC_1000k_stage2_cellflow.sh full 50 20260805_final
# Arguments: mode, maximum epochs, the same run tag used for Stage 1.

source /home1/10119/ghzheng/.bashrc
conda activate worldmodeltraining

cd /scratch/10119/ghzheng/Virtual_microenv
set -e

MODE="${1:-full}"
RUN_TAG="${3:-$(date +%Y%m%d)}"
case "$MODE" in
  test)
    CELLFLOW_EPOCHS=1
    CELLFLOW_MAX_STEPS=10
    VALIDATE_EVERY=1
    MAX_VAL_STEPS=2
    RUN_TAG="${RUN_TAG}_test"
    ;;
  full)
    CELLFLOW_EPOCHS="${2:-50}"
    CELLFLOW_MAX_STEPS=-1
    VALIDATE_EVERY=5
    MAX_VAL_STEPS=100
    ;;
  *)
    echo "ERROR: mode must be test or full"
    exit 2
    ;;
esac

python train_five_level_world_model.py --help | grep -q -- "--validate-every-epochs" || {
  echo "ERROR: train_five_level_world_model.py is not the revised CellFlow trainer"
  exit 1
}

test -d paired_training_h5ad_1000k_fast || {
  echo "ERROR: missing paired_training_h5ad_1000k_fast"
  exit 1
}

VAE_DIR="stage1_${RUN_TAG}_gaussian_vae_1000k"
VAE_CHECKPOINT="$VAE_DIR/best_stage1_vae_fold0.pt"
CELLFLOW_DIR="stage2_${RUN_TAG}_vae_cellflow_drift3_medium_pop_1000k"

test -f "$VAE_CHECKPOINT" || {
  echo "ERROR: Stage-1 VAE checkpoint is missing: $VAE_CHECKPOINT"
  echo "Run TACC_1000k_stage1_gaussian_vae.sh first."
  exit 1
}

echo "============================================================"
echo "Stage 2 only: frozen VAE plus 1M CellFlow"
echo "python=$(command -v python)"
echo "mode=$MODE"
echo "run_tag=$RUN_TAG"
echo "maximum_epochs=$CELLFLOW_EPOCHS"
echo "stage1_checkpoint=$VAE_CHECKPOINT"
echo "output=$CELLFLOW_DIR"
echo "loss=flow + 0.1*endpoint + 3e-4*MMD + 3e-3*W2 + 3.0*drift"
echo "============================================================"

python train_five_level_world_model.py \
  --data-dir paired_training_h5ad_1000k_fast \
  --stage1-vae-checkpoint "$VAE_CHECKPOINT" \
  --representation-type gaussian_vae \
  --expression-transform log1p_10k \
  --latent-dim 128 \
  --expression-hidden-dim 512 \
  --expression-depth 3 \
  --model-dim 256 \
  --depth 4 \
  --heads 8 \
  --num-tokens 8 \
  --level 6 \
  --loss-mode cellflow \
  --loss-tag vae_cellflow_drift_3_medium_pop \
  --batch-size 2048 \
  --epochs "$CELLFLOW_EPOCHS" \
  --max-steps "$CELLFLOW_MAX_STEPS" \
  --num-folds 1 \
  --fold-index 0 \
  --lr 1e-5 \
  --save-every 4890 \
  --no-save-each-epoch \
  --validate-every-epochs "$VALIDATE_EVERY" \
  --max-val-steps "$MAX_VAL_STEPS" \
  --sinkhorn-iters 20 \
  --cellflow-sigma 0.01 \
  --cellflow-integration-steps 8 \
  --cellflow-integration-method euler \
  --cellflow-pair-aux-weight 0.1 \
  --cellflow-population-aux-weight 1.0 \
  --lambda-mmd 3e-4 \
  --lambda-w2 3e-3 \
  --lambda-drift 3.0 \
  --lambda-down 0.0 \
  --checkpoint-dir "$CELLFLOW_DIR" \
  --device cuda

FINAL_CHECKPOINT="$CELLFLOW_DIR/best_model_level6_fold0.pt"
RUN_CONFIG="$CELLFLOW_DIR/run_config.json"

test -f "$FINAL_CHECKPOINT" || {
  echo "ERROR: CellFlow checkpoint was not created: $FINAL_CHECKPOINT"
  exit 1
}

test -f "$RUN_CONFIG" || {
  echo "ERROR: run_config.json was not created: $RUN_CONFIG"
  exit 1
}

grep -q '"representation_type": "gaussian_vae"' "$RUN_CONFIG" || {
  echo "ERROR: output is not the Gaussian-VAE model"
  exit 1
}

echo "============================================================"
echo "Stage 2 finished"
echo "best_checkpoint=$FINAL_CHECKPOINT"
echo "============================================================"
