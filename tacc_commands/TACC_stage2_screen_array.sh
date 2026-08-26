#!/bin/bash
#SBATCH -J vm_s2_screen
#SBATCH -o logs/s2_screen_%A_%a.out
#SBATCH -e logs/s2_screen_%A_%a.err
#SBATCH -p gh
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -a 0-12%4
#SBATCH --mail-user=genhuizheng@utexas.edu
#SBATCH --mail-type=FAIL,END
#SBATCH -t 12:00:00
#SBATCH -A MCB26031

# Stage-2 architecture screen: one transition model per array task, all on the
# SAME frozen Stage-1 representation.
#
# This is the comparison the earlier 50k/500k rounds could not make. Those runs
# trained a separate encoder per architecture with no reconstruction
# constraint, so each level shaped a latent that made its own task easy and the
# rankings partly measured coordinate choice rather than dynamics. With one
# frozen encoder shared by every level, a difference here is a difference in
# how well the model captures the pre->post move.
#
# Levels 0-6 plus every level-5 variant. `stochastic_dit` is omitted because it
# maps to the same class as `default` (AdvancedDiTResidual) in
# five_level_cell_world_model.py -- running both would double-count one model.
#
# Level 0 is the non-learned identity baseline: it predicts no change and has
# no trainable transition parameters. Any level that fails to beat it has not
# learned a treatment effect at all, which is the single most important thing
# this screen can tell you.
#
# Comparability note: every config uses --loss-mode pair_mse so all are scored
# on one objective (val_loss_pair). level6_cellflow_fm additionally runs
# CellFlow's native flow-matching objective; its val_loss_pair is still logged
# and comparable, but its checkpoint selection uses the total loss instead.
#
# Adjust -a to match the config count:
#     tail -n +2 stage2_screen_configs.tsv | wc -l
#
# Usage:
#   sbatch tacc_commands/TACC_stage2_screen_array.sh
#   sbatch tacc_commands/TACC_stage2_screen_array.sh stage2_screen_configs.tsv 0 50
#
# Rank when finished:
#   python rank_stage2_screen.py --pattern 'checkpoints_s2_*'

set -eo pipefail
task_started=$SECONDS
# Set VM_SKIP_ENV=1 to use the already-active environment (local testing, or
# when the caller has activated it). Sourcing .bashrc is non-fatal: an
# unrelated failure in a shell profile should not take down a compute job.
if [[ -z "${VM_SKIP_ENV:-}" ]]; then
  [[ -f /home1/10119/ghzheng/.bashrc ]] && source /home1/10119/ghzheng/.bashrc || true
  CONDA_ENV="${VM_CONDA_ENV:-/scratch/10119/ghzheng/conda_envs/worldmodel_withconfidenceot}"
  conda activate "$CONDA_ENV" || { echo "ERROR: cannot activate $CONDA_ENV"; exit 1; }
  echo "conda_env=$CONDA_ENV"
fi
echo "python=$(command -v python)"

cd "${SLURM_SUBMIT_DIR:-$(dirname "$(readlink -f "$0")")/..}"

CONFIGS="${1:-stage2_screen_configs.tsv}"
FOLD="${2:-0}"
EPOCHS="${3:-50}"
DATA_DIR="${4:-paired_training_h5ad_50k}"
# The Stage-1 representation chosen after the fold-0 screen and 5-fold CV:
# best treatment-signal retention (role 1.44x) of anything tested, and the only
# option that encodes a new dataset with no batch label. See
# docs/HANDOFF_20260826.md section 7.
STAGE1="${5:-stage1_screen_p90_nob_nb/best_stage1_scvi_fold0.pt}"
LOOKUP="${6:-../Virtual_microenv/dataset_platform_lookup.tsv}"

mkdir -p logs
test -f "$CONFIGS" || { echo "ERROR: missing $CONFIGS"; exit 1; }
test -d "$DATA_DIR" || { echo "ERROR: missing $DATA_DIR"; exit 1; }
test -f "$STAGE1" || { echo "ERROR: missing Stage-1 checkpoint: $STAGE1"; exit 1; }

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

CKPT_DIR="checkpoints_s2_${TAG}_fold${FOLD}"

echo "============================================================"
echo "task=$IDX  tag=$TAG"
echo "  level=$LEVEL variant=$VARIANT loss_mode=$LOSS_MODE fold=$FOLD"
echo "  stage1=$STAGE1"
echo "============================================================"

if [[ -f "$CKPT_DIR/val_epoch_losses.csv" ]]; then
  echo "already complete -> $CKPT_DIR (delete it to force a rerun)"
  echo "TASK_WALL_SECONDS=$((SECONDS-task_started))"
  exit 0
fi

# Level 0 has no trainable parameters, so it runs a single evaluation pass.
RUN_EPOCHS="$EPOCHS"
if [[ "$LEVEL" == "0" ]]; then
  RUN_EPOCHS=1
fi

# --expression-transform and --latent-dim are deliberately omitted: Stage 2
# adopts both from the Stage-1 checkpoint, along with the batch key, gene
# panel and technology filter, so the two stages cannot disagree about what
# the encoder expects or what latent the transition lives in.
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
