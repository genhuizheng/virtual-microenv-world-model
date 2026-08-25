#!/bin/bash
#SBATCH -J vm_1m_vae_cellflow
#SBATCH -o vm_1m_vae_cellflow_%j.out
#SBATCH -e vm_1m_vae_cellflow_%j.err
#SBATCH -p gh
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --mail-user=genhuizheng@utexas.edu
#SBATCH --mail-type=all
#SBATCH -t 48:00:00
#SBATCH -A MCB26031

# Corrected 1M model:
#   raw counts on disk -> log1p_10k -> trained Gaussian VAE -> frozen latent
#   -> Level-6 CellFlow with the previously selected drift_3_medium_pop loss.
source /home1/10119/ghzheng/.bashrc
conda activate worldmodeltraining

cd /scratch/10119/ghzheng/Virtual_microenv
set -e

MODE="${1:-full}"
case "$MODE" in
  test)
    STAGE1_EPOCHS=1
    STAGE2_EPOCHS=1
    STAGE1_LIMIT_ARGS=(--max-train-steps 5 --max-val-steps 2)
    STAGE2_MAX_STEPS=10
    STAGE2_VALIDATE_EVERY=1
    STAGE2_MAX_VAL_STEPS=2
    OUTPUT_SUFFIX="_test"
    ;;
  full)
    STAGE1_EPOCHS=2
    STAGE2_EPOCHS=50
    STAGE1_LIMIT_ARGS=(--max-val-steps 100)
    STAGE2_MAX_STEPS=-1
    STAGE2_VALIDATE_EVERY=5
    STAGE2_MAX_VAL_STEPS=100
    OUTPUT_SUFFIX=""
    ;;
  *)
    echo "ERROR: mode must be test or full"
    exit 2
    ;;
esac

python train_five_level_world_model.py --help | grep -q -- "--stage1-vae-checkpoint" || {
  echo "ERROR: train_five_level_world_model.py is not the corrected VAE trainer"
  exit 1
}

python train_five_level_world_model.py --help | grep -q -- "--cellflow-sigma" || {
  echo "ERROR: train_five_level_world_model.py is not the complete CellFlow trainer"
  exit 1
}

test -f train_stage1_gaussian_vae.py || {
  echo "ERROR: missing train_stage1_gaussian_vae.py"
  exit 1
}

test -d paired_training_h5ad_1000k_fast || {
  echo "ERROR: missing paired_training_h5ad_1000k_fast"
  exit 1
}

STAGE1_DIR="checkpoints_stage1_gaussian_vae_1000k_full_data${OUTPUT_SUFFIX}"
STAGE1_CHECKPOINT="$STAGE1_DIR/best_stage1_vae_fold0.pt"
STAGE2_DIR="checkpoints_1000k_vae_cellflow_drift3_medium_pop_full_data${OUTPUT_SUFFIX}"

echo "python=$(command -v python)"
echo "mode=$MODE"
echo "stage1=$STAGE1_CHECKPOINT"
echo "stage2=$STAGE2_DIR"

echo "============================================================"
echo "Stage 1: train Gaussian VAE decoder and encoder"
echo "mode=$MODE"
echo "output=$STAGE1_DIR"
echo "============================================================"

python train_stage1_gaussian_vae.py \
  --data-dir paired_training_h5ad_1000k_fast \
  --checkpoint-dir "$STAGE1_DIR" \
  --epochs "$STAGE1_EPOCHS" \
  --batch-size 4096 \
  --lr 1e-3 \
  --kl-weight 1e-3 \
  --latent-dim 128 \
  --hidden-dim 512 \
  --depth 3 \
  --num-folds 1 \
  --fold-index 0 \
  "${STAGE1_LIMIT_ARGS[@]}" \
  --device cuda \
  --overwrite

test -f "$STAGE1_CHECKPOINT" || {
  echo "ERROR: Stage-1 checkpoint was not created: $STAGE1_CHECKPOINT"
  exit 1
}

echo "============================================================"
echo "Stage 2: frozen VAE plus 1M CellFlow"
echo "loss=flow + 0.1*endpoint + 3e-4*MMD + 3e-3*W2 + 3.0*drift"
echo "output=$STAGE2_DIR"
echo "============================================================"

python train_five_level_world_model.py \
  --data-dir paired_training_h5ad_1000k_fast \
  --stage1-vae-checkpoint "$STAGE1_CHECKPOINT" \
  --representation-type gaussian_vae \
  --expression-transform log1p_10k \
  --expression-hidden-dim 512 \
  --expression-depth 3 \
  --level 6 \
  --loss-mode cellflow \
  --loss-tag vae_cellflow_drift_3_medium_pop \
  --batch-size 2048 \
  --epochs "$STAGE2_EPOCHS" \
  --max-steps "$STAGE2_MAX_STEPS" \
  --num-folds 1 \
  --fold-index 0 \
  --lr 1e-5 \
  --save-every 4890 \
  --no-save-each-epoch \
  --validate-every-epochs "$STAGE2_VALIDATE_EVERY" \
  --max-val-steps "$STAGE2_MAX_VAL_STEPS" \
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
  --checkpoint-dir "$STAGE2_DIR" \
  --device cuda

FINAL_CHECKPOINT="$STAGE2_DIR/best_model_level6_fold0.pt"
RUN_CONFIG="$STAGE2_DIR/run_config.json"

test -f "$FINAL_CHECKPOINT" || {
  echo "ERROR: corrected final checkpoint was not created: $FINAL_CHECKPOINT"
  exit 1
}

test -f "$RUN_CONFIG" || {
  echo "ERROR: run_config.json was not created: $RUN_CONFIG"
  exit 1
}

grep -q '"representation_type": "gaussian_vae"' "$RUN_CONFIG" || {
  echo "ERROR: output checkpoint is not gaussian_vae"
  exit 1
}

grep -q '"expression_transform": "log1p_10k"' "$RUN_CONFIG" || {
  echo "ERROR: output checkpoint does not use log1p_10k"
  exit 1
}

grep -q '"loss_tag": "vae_cellflow_drift_3_medium_pop"' "$RUN_CONFIG" || {
  echo "ERROR: output checkpoint has the wrong loss tag"
  exit 1
}

echo "============================================================"
echo "Corrected Gaussian-VAE plus CellFlow training finished"
echo "checkpoint=$FINAL_CHECKPOINT"
echo "============================================================"
