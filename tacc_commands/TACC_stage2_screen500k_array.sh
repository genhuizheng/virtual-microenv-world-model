#!/bin/bash
#SBATCH -J vm_s2_500k
#SBATCH -o logs/s2_500k_%A_%a.out
#SBATCH -e logs/s2_500k_%A_%a.err
#SBATCH -p gh
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -a 0-20%4
#SBATCH --mail-user=genhuizheng@utexas.edu
#SBATCH --mail-type=FAIL,END
#SBATCH -t 12:00:00
#SBATCH -A MCB26031

# Stage-2 architecture x objective screen on the 500k pair set.
#
# 10 architectures x 2 objectives, plus CellFlow's native flow-matching run.
#
# WHAT CHANGED FROM THE 50k SCREEN, AND WHY
#
# 1. The objective is now an axis, not a constant. At 50k every config was
#    trained on pair_mse and they clustered inside 1.7 percentage points, all
#    bottoming out by epoch 0-1. The diagnostic was that metric_mmd got WORSE
#    while loss_pair improved -- even during epochs when val loss_pair was
#    still genuinely falling on held-out patients. That is regression to the
#    mean: MOSCOT gives a probabilistic coupling, not a ground-truth cell-to-
#    cell correspondence, so the per-cell target carries irreducible pairing
#    noise and minimising MSE against it shrinks the predicted distribution.
#    --loss-mode chreode optimises MMD + Sinkhorn W2 + drift instead.
#    Both metrics are logged in every mode, so the two arms stay comparable.
#
# 2. tokenized_dit and rectified_dit are GONE. They were provably identical to
#    levels 3 and 2 -- byte-identical val curves to 16 decimals and matching
#    parameter counts. DiTTokenBackbone.forward computes the same expression as
#    DiTBackbone.forward, and with delta constant at 1.0 the rectified variant's
#    `step * velocity` reduces to plain `residual`. Running them again would
#    burn two GPU-hours to re-derive a tie.
#
# 3. Epochs: 10, not 50. At 500k an epoch is ~1,560 steps instead of ~157, so
#    this is ~15,600 steps against the 50k screen's 7,850 -- twice the
#    optimisation in a fifth of the epochs. The 50k curves turned after 1-2
#    passes over the data, and passes are what drive overfitting.
#
# CAVEAT WORTH KNOWING: the 500k set has 408 patients against 50k's 406. It is
# ~10x more cells from the SAME patients, not more patient diversity. It should
# help the model learn cell-state-conditional dynamics (pairs per patient goes
# from ~123 to ~1,230) rather than per-patient mean shifts, but it will not on
# its own close the cross-patient generalisation gap.
#
# Level 0 is the floor in BOTH arms and is the most important number here. For
# pair_mse it is the loss of predicting no change. For chreode it is the MMD/W2
# between the pre and post distributions as they stand -- a much harder floor,
# because an identity map already has decent population overlap. A config that
# beats level 0 on pair_mse but not on chreode has learned a mean shift.
#
# Usage:
#   sbatch tacc_commands/TACC_stage2_screen500k_array.sh
#   sbatch tacc_commands/TACC_stage2_screen500k_array.sh stage2_screen500k_configs.tsv 0 10
#
# Rank when finished:
#   python rank_stage2_screen.py --pattern 'checkpoints_s2_500k_*'

set -eo pipefail
task_started=$SECONDS
if [[ -z "${VM_SKIP_ENV:-}" ]]; then
  [[ -f /home1/10119/ghzheng/.bashrc ]] && source /home1/10119/ghzheng/.bashrc || true
  CONDA_ENV="${VM_CONDA_ENV:-/scratch/10119/ghzheng/conda_envs/worldmodel_withconfidenceot}"
  conda activate "$CONDA_ENV" || { echo "ERROR: cannot activate $CONDA_ENV"; exit 1; }
  echo "conda_env=$CONDA_ENV"
fi
echo "python=$(command -v python)"

cd "${SLURM_SUBMIT_DIR:-$(dirname "$(readlink -f "$0")")/..}"

CONFIGS="${1:-stage2_screen500k_configs.tsv}"
FOLD="${2:-0}"
EPOCHS="${3:-10}"
DATA_DIR="${4:-paired_training_h5ad_500k_fast}"
# MUST be the 500k Stage-1 refit, not the 50k checkpoint. Fold membership is
# now hash-based (stable across data directories), but every checkpoint trained
# before that fix used positional assignment and therefore sits on a different
# split. See tacc_commands/TACC_stage1_500k.sh.
STAGE1="${5:-stage1_500k_p90_nob_nb/best_stage1_scvi_fold0.pt}"
LOOKUP="${6:-../Virtual_microenv/dataset_platform_lookup.tsv}"

mkdir -p logs
test -f "$CONFIGS" || { echo "ERROR: missing $CONFIGS"; exit 1; }
test -d "$DATA_DIR" || { echo "ERROR: missing $DATA_DIR"; exit 1; }
test -f "$STAGE1" || {
  echo "ERROR: missing Stage-1 checkpoint: $STAGE1"
  echo "       Run tacc_commands/TACC_stage1_500k.sh first."
  echo "       Do NOT substitute the 50k checkpoint: it was trained under the"
  echo "       old positional fold map and leaks 68 of ~82 fold-0 val patients."
  exit 1
}

IDX="${SLURM_ARRAY_TASK_ID:-0}"

row=$(python - "$CONFIGS" "$IDX" <<'PY'
import csv, sys
with open(sys.argv[1], newline="") as handle:
    rows = [r for r in csv.DictReader(handle, delimiter="\t") if r.get("tag")]
idx = int(sys.argv[2])
if idx >= len(rows):
    sys.exit(f"array index {idx} exceeds {len(rows)} configurations")
r = rows[idx]
print("\t".join(r[k] for k in ("tag", "level", "variant", "loss_mode")))
PY
)
IFS=$'\t' read -r TAG LEVEL VARIANT LOSS_MODE <<< "$row"

CKPT_DIR="checkpoints_s2_500k_${TAG}_fold${FOLD}"

echo "============================================================"
echo "task=$IDX  tag=$TAG"
echo "  level=$LEVEL variant=$VARIANT loss_mode=$LOSS_MODE fold=$FOLD epochs=$EPOCHS"
echo "  stage1=$STAGE1"
echo "  data=$DATA_DIR"
echo "============================================================"

if [[ -f "$CKPT_DIR/val_epoch_losses.csv" ]]; then
  echo "already complete -> $CKPT_DIR (delete it to force a rerun)"
  echo "TASK_WALL_SECONDS=$((SECONDS-task_started))"
  exit 0
fi

# Level 0 has no trainable parameters, so one pass measures the floor exactly.
RUN_EPOCHS="$EPOCHS"
if [[ "$LEVEL" == "0" ]]; then
  RUN_EPOCHS=1
fi

# --latent-dim and --expression-transform are omitted deliberately: Stage 2
# adopts them from the Stage-1 checkpoint along with the batch key, gene panel
# and technology filter, so the two stages cannot disagree.
python train_five_level_world_model.py \
  --data-dir "$DATA_DIR" \
  --level "$LEVEL" \
  --variant "$VARIANT" \
  --loss-mode "$LOSS_MODE" \
  --loss-tag "$TAG" \
  --stage1-scvi-checkpoint "$STAGE1" \
  --technology-lookup "$LOOKUP" \
  --group-column patient_id --num-folds 5 --fold-index "$FOLD" \
  --batch-size 256 --epochs "$RUN_EPOCHS" \
  --model-dim 256 --depth 4 --heads 8 --num-tokens 8 \
  --checkpoint-dir "$CKPT_DIR" \
  --device cuda

echo "COMPLETE $TAG -> $CKPT_DIR"
echo "TASK_WALL_SECONDS=$((SECONDS-task_started))"
