#!/bin/bash
#SBATCH -J vm_s1_screen
#SBATCH -o logs/s1_screen_%A_%a.out
#SBATCH -e logs/s1_screen_%A_%a.err
#SBATCH -p gh
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -a 0-19%4
#SBATCH --mail-user=genhuizheng@utexas.edu
#SBATCH --mail-type=FAIL,END
#SBATCH -t 08:00:00
#SBATCH -A MCB26031

# One Stage-1 configuration per array task, read by index from a TSV.
#
# Adjust -a to match the config count: `0-N%K` runs N+1 tasks, K at a time.
# The default 0-19%4 matches the 20-config factorial in
# stage1_screen_configs.tsv. Check with:
#     tail -n +2 stage1_screen_configs.tsv | wc -l
#
# Each task trains one configuration and evaluates it, writing
# stage1_screen_<tag>/ and eval_screen_<tag>_fold<N>.json. Tasks are
# independent, so one failure does not take down the screen, and completed
# tasks are skipped on resubmission.
#
# Usage:
#   sbatch tacc_commands/TACC_stage1_screen_array.sh
#   sbatch tacc_commands/TACC_stage1_screen_array.sh stage1_screen_configs.tsv 0 50
#
# Rank once the array finishes:
#   python rank_stage1_screen.py --pattern 'eval_screen_*_fold0.json'

set -eo pipefail
task_started=$SECONDS
source /home1/10119/ghzheng/.bashrc
conda activate worldmodeltraining

cd "${SLURM_SUBMIT_DIR:-$(dirname "$(readlink -f "$0")")/..}"

CONFIGS="${1:-stage1_screen_configs.tsv}"
FOLD="${2:-0}"
MAX_EPOCHS="${3:-50}"
DATA_DIR="${4:-paired_training_h5ad_50k}"
LOOKUP="${5:-../Virtual_microenv/dataset_platform_lookup.tsv}"
GENE_DIR="${6:-../Virtual_microenv}"

mkdir -p logs

test -f "$CONFIGS" || { echo "ERROR: missing $CONFIGS"; exit 1; }
test -d "$DATA_DIR" || { echo "ERROR: missing $DATA_DIR"; exit 1; }
test -f "$LOOKUP"  || { echo "ERROR: missing $LOOKUP"; exit 1; }

IDX="${SLURM_ARRAY_TASK_ID:-0}"

# Pull this task's configuration row. Fails loudly if the array range exceeds
# the config count, rather than silently running nothing.
row=$(python - "$CONFIGS" "$IDX" <<'PY'
import csv, sys
with open(sys.argv[1], newline="") as handle:
    rows = [r for r in csv.DictReader(handle, delimiter="\t") if r.get("tag")]
idx = int(sys.argv[2])
if idx >= len(rows):
    sys.exit(f"array index {idx} exceeds {len(rows)} configurations")
r = rows[idx]
print("\t".join(r[k] for k in (
    "tag", "gene_likelihood", "expression_transform", "latent_dim",
    "n_hidden", "n_layers", "dispersion", "batch_key", "gene_list")))
PY
)
IFS=$'\t' read -r TAG LIK XFORM LATENT HIDDEN LAYERS DISP BATCHKEY GENELIST <<< "$row"

case "$GENELIST" in
  90pct)   GENE_ARG=(--keep-gene-list "$GENE_DIR/genes_measured_90pct.txt") ;;
  50pct)   GENE_ARG=(--keep-gene-list "$GENE_DIR/genes_measured_50pct.txt") ;;
  none|"") GENE_ARG=() ;;
  *)       GENE_ARG=(--keep-gene-list "$GENE_DIR/$GENELIST") ;;
esac

CKPT_DIR="stage1_screen_${TAG}"
CKPT="$CKPT_DIR/best_stage1_scvi_fold${FOLD}.pt"
EVAL_OUT="eval_screen_${TAG}_fold${FOLD}.json"

echo "============================================================"
echo "task=$IDX  tag=$TAG"
echo "  likelihood=$LIK transform=$XFORM latent=$LATENT hidden=$HIDDEN layers=$LAYERS"
echo "  dispersion=$DISP batch_key=$BATCHKEY genes=$GENELIST fold=$FOLD"
echo "============================================================"

if [[ -f "$EVAL_OUT" ]]; then
  echo "already complete -> $EVAL_OUT (delete it to force a rerun)"
  echo "TASK_WALL_SECONDS=$((SECONDS-task_started))"
  exit 0
fi

python train_stage1_scvi.py \
  --data-dir "$DATA_DIR" \
  --checkpoint-dir "$CKPT_DIR" \
  --max-epochs "$MAX_EPOCHS" \
  --batch-size 256 \
  --latent-dim "$LATENT" \
  --n-hidden "$HIDDEN" \
  --n-layers "$LAYERS" \
  --dispersion "$DISP" \
  --gene-likelihood "$LIK" \
  --expression-transform "$XFORM" \
  --num-folds 5 --fold-index "$FOLD" \
  --group-column patient_id \
  --batch-key "$BATCHKEY" \
  "${GENE_ARG[@]}" \
  --min-cells-detected 1 \
  --technology-lookup "$LOOKUP" \
  --keep-technologies "10x genomics" \
  --device cuda --overwrite

test -f "$CKPT" || { echo "ERROR: checkpoint not written: $CKPT"; exit 1; }

python evaluate_stage1_scvi.py \
  --checkpoint "$CKPT" \
  --data-dir "$DATA_DIR" \
  --technology-lookup "$LOOKUP" \
  --num-folds 5 --fold-index "$FOLD" \
  --max-cells 20000 \
  --device cuda \
  --out "$EVAL_OUT"

echo "COMPLETE $TAG -> $EVAL_OUT"
echo "TASK_WALL_SECONDS=$((SECONDS-task_started))"
