#!/bin/bash
#SBATCH -J loss_pop_500k
#SBATCH -o loss_pop_500k_%j.out
#SBATCH -e loss_pop_500k_%j.err
#SBATCH -p gh
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --mail-user=genhuizheng@utexas.edu
#SBATCH --mail-type=all
#SBATCH -t 48:00:00
#SBATCH -A MCB26031

source /home1/10119/ghzheng/.bashrc
conda activate worldmodeltraining

cd /scratch/10119/ghzheng/Virtual_microenv

DATA_DIR="paired_training_h5ad_500k_fast"
FOLD=0
EPOCHS=50
BATCH_SIZE=2048
LR="1e-5"
SINKHORN_ITERS=20

# tag lambda_mmd lambda_w2 lambda_drift lambda_down
settings=(
  "pair_anchor 0.0 0.0 1.0 0.0"
  "pop_1e6_1e5 1e-6 1e-5 1.0 0.0"
  "pop_3e6_3e5 3e-6 3e-5 1.0 0.0"
  "pop_1e5_1e4 1e-5 1e-4 1.0 0.0"
  "pop_3e5_3e4 3e-5 3e-4 1.0 0.0"
  "pop_1e4_1e3 1e-4 1e-3 1.0 0.0"
  "pop_3e4_3e3 3e-4 3e-3 1.0 0.0"
  "pop_1e3_1e2 1e-3 1e-2 1.0 0.0"
  "pop_3e3_3e2 3e-3 3e-2 1.0 0.0"
  "pop_mmd_low_w2_mid 1e-5 1e-3 1.0 0.0"
  "pop_mmd_mid_w2_low 1e-4 1e-4 1.0 0.0"
  "pop_mmd_mid_w2_high 1e-4 1e-2 1.0 0.0"
  "pop_mmd_high_w2_mid 1e-3 1e-3 1.0 0.0"
)

for setting in "${settings[@]}"; do
  read -r tag lambda_mmd lambda_w2 lambda_drift lambda_down <<< "$setting"
  echo "===== 500k Chreode loss screen POP: ${tag}, fold=${FOLD} ====="
  python train_five_level_world_model.py \
    --data-dir "${DATA_DIR}" \
    --level 5 \
    --variant hierarchical_dit \
    --loss-mode chreode \
    --loss-tag "${tag}" \
    --lambda-drift "${lambda_drift}" \
    --lambda-mmd "${lambda_mmd}" \
    --lambda-w2 "${lambda_w2}" \
    --lambda-down "${lambda_down}" \
    --batch-size "${BATCH_SIZE}" \
    --epochs "${EPOCHS}" \
    --max-steps -1 \
    --num-folds 5 \
    --fold-index "${FOLD}" \
    --lr "${LR}" \
    --save-every 0 \
    --no-save-each-epoch \
    --sinkhorn-iters "${SINKHORN_ITERS}" \
    --checkpoint-dir "checkpoints_500k_loss_screen_level5_hierarchical_dit_${tag}_fold${FOLD}"
done

python rank_500k_loss_screen.py \
  --glob "checkpoints_500k_loss_screen_level5_hierarchical_dit_*_fold*/val_epoch_losses.csv" \
  --out-csv "500k_loss_screen_fold0_summary.csv"
