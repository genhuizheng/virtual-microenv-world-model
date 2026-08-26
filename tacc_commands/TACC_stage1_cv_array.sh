#!/bin/bash
#SBATCH -J vm_s1_cv
#SBATCH -o logs/s1_cv_%A_%a.out
#SBATCH -e logs/s1_cv_%A_%a.err
#SBATCH -p gh
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -a 0-11%4
#SBATCH --mail-user=genhuizheng@utexas.edu
#SBATCH --mail-type=FAIL,END
#SBATCH -t 08:00:00
#SBATCH -A MCB26031

# Cross-validate the configs promoted from the fold-0 screen.
#
# One array task per (config, fold) pair. Tags come from stage1_promote.txt
# and their hyperparameters are looked up by tag in stage1_screen_configs.tsv,
# so there is a single source of truth for what each configuration means.
#
# Outputs reuse the screen's naming -- stage1_screen_<tag>/ and
# eval_screen_<tag>_fold<N>.json -- so fold 0 from the screen combines with
# these automatically, and rank_stage1_screen.py averages across folds when
# given a pattern spanning them:
#     python rank_stage1_screen.py --pattern 'eval_screen_*_fold*.json'
#
# Array size must be n_tags * n_folds. With 3 tags and folds "1 2 3 4" that is
# 12, hence 0-11%4. Check with:
#     grep -vc '^#\|^$' stage1_promote.txt
#
# Usage:
#   sbatch tacc_commands/TACC_stage1_cv_array.sh
#   sbatch tacc_commands/TACC_stage1_cv_array.sh stage1_promote.txt "1 2 3 4" 50

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

PROMOTE="${1:-stage1_promote.txt}"
FOLDS="${2:-1 2 3 4}"
MAX_EPOCHS="${3:-50}"
CONFIGS="${4:-stage1_screen_configs.tsv}"
DATA_DIR="${5:-paired_training_h5ad_50k}"
LOOKUP="${6:-../Virtual_microenv/dataset_platform_lookup.tsv}"
GENE_DIR="${7:-../Virtual_microenv}"
MASK_NPZ="${8:-../Virtual_microenv/gene_masks_by_dataset.npz}"

mkdir -p logs
test -f "$PROMOTE" || { echo "ERROR: missing $PROMOTE"; exit 1; }
test -f "$CONFIGS" || { echo "ERROR: missing $CONFIGS"; exit 1; }
test -d "$DATA_DIR" || { echo "ERROR: missing $DATA_DIR"; exit 1; }

# Map this task to a (tag, fold) pair.
mapfile -t TAGS < <(grep -v '^#' "$PROMOTE" | grep -v '^[[:space:]]*$' | tr -d '\r')
read -r -a FOLD_ARR <<< "$FOLDS"
N_TAGS=${#TAGS[@]}
N_FOLDS=${#FOLD_ARR[@]}
TOTAL=$((N_TAGS * N_FOLDS))
IDX="${SLURM_ARRAY_TASK_ID:-0}"

if (( IDX >= TOTAL )); then
  echo "ERROR: task $IDX exceeds $TOTAL (= $N_TAGS tags x $N_FOLDS folds)."
  echo "       Set '#SBATCH -a 0-$((TOTAL-1))%4' to match."
  exit 1
fi

TAG="${TAGS[$((IDX / N_FOLDS))]}"
FOLD="${FOLD_ARR[$((IDX % N_FOLDS))]}"

# Look this tag's hyperparameters up in the screen config, so the CV run and
# the screen run cannot drift apart.
row=$(python - "$CONFIGS" "$TAG" <<'PY'
import csv, sys
with open(sys.argv[1], newline="") as handle:
    rows = {r["tag"]: r for r in csv.DictReader(handle, delimiter="\t") if r.get("tag")}
tag = sys.argv[2]
if tag not in rows:
    sys.exit(f"tag {tag!r} not found in {sys.argv[1]}")
r = rows[tag]
print("\t".join(r[k] for k in (
    "trainer", "use_mask", "gene_likelihood", "expression_transform",
    "latent_dim", "n_hidden", "n_layers", "dispersion", "batch_key", "gene_list")))
PY
)
IFS=$'\t' read -r TRAINER USE_MASK LIK XFORM LATENT HIDDEN LAYERS DISP BATCHKEY GENELIST <<< "$row"

case "$GENELIST" in
  90pct)   GENE_ARG=(--keep-gene-list "$GENE_DIR/genes_measured_90pct.txt") ;;
  50pct)   GENE_ARG=(--keep-gene-list "$GENE_DIR/genes_measured_50pct.txt") ;;
  none|"") GENE_ARG=() ;;
  *)       GENE_ARG=(--keep-gene-list "$GENE_DIR/$GENELIST") ;;
esac

CKPT_DIR="stage1_screen_${TAG}"
EVAL_OUT="eval_screen_${TAG}_fold${FOLD}.json"

echo "============================================================"
echo "task=$IDX  tag=$TAG  fold=$FOLD  ($N_TAGS tags x $N_FOLDS folds)"
echo "  trainer=$TRAINER likelihood=$LIK transform=$XFORM"
echo "  latent=$LATENT hidden=$HIDDEN layers=$LAYERS batch_key=$BATCHKEY genes=$GENELIST"
echo "============================================================"

if [[ -f "$EVAL_OUT" ]]; then
  echo "already complete -> $EVAL_OUT (delete it to force a rerun)"
  echo "TASK_WALL_SECONDS=$((SECONDS-task_started))"
  exit 0
fi

case "$TRAINER" in
  scvi)
    CKPT="$CKPT_DIR/best_stage1_scvi_fold${FOLD}.pt"
    python train_stage1_scvi.py \
      --data-dir "$DATA_DIR" --checkpoint-dir "$CKPT_DIR" \
      --max-epochs "$MAX_EPOCHS" --batch-size 256 \
      --latent-dim "$LATENT" --n-hidden "$HIDDEN" --n-layers "$LAYERS" \
      --dispersion "$DISP" --gene-likelihood "$LIK" \
      --expression-transform "$XFORM" \
      --num-folds 5 --fold-index "$FOLD" --group-column patient_id \
      --batch-key "$BATCHKEY" "${GENE_ARG[@]}" --min-cells-detected 1 \
      --technology-lookup "$LOOKUP" --keep-technologies "10x genomics" \
      --device cuda --overwrite
    ;;
  masked)
    CKPT="$CKPT_DIR/best_stage1_masked_vae_fold${FOLD}.pt"
    MASK_ARG=()
    if [[ "$USE_MASK" == "yes" ]]; then
      test -f "$MASK_NPZ" || { echo "ERROR: missing gene mask npz: $MASK_NPZ"; exit 1; }
      MASK_ARG=(--gene-mask-npz "$MASK_NPZ")
    fi
    python train_stage1_masked_vae.py \
      --data-dir "$DATA_DIR" --checkpoint-dir "$CKPT_DIR" "${MASK_ARG[@]}" \
      --max-epochs "$MAX_EPOCHS" --batch-size 256 \
      --latent-dim "$LATENT" --n-hidden "$HIDDEN" --n-layers "$LAYERS" \
      --expression-transform "$XFORM" \
      --num-folds 5 --fold-index "$FOLD" --group-column patient_id \
      --batch-key "$BATCHKEY" "${GENE_ARG[@]}" --min-cells-detected 1 \
      --technology-lookup "$LOOKUP" --keep-technologies "10x genomics" \
      --device cuda --overwrite
    ;;
  *)
    echo "ERROR: unknown trainer '$TRAINER' for tag '$TAG'"; exit 2 ;;
esac

test -f "$CKPT" || { echo "ERROR: checkpoint not written: $CKPT"; exit 1; }

EVAL_MASK_ARG=()
if [[ -f "$MASK_NPZ" ]]; then
  EVAL_MASK_ARG=(--gene-mask-npz "$MASK_NPZ")
fi

python evaluate_stage1_scvi.py \
  --checkpoint "$CKPT" --data-dir "$DATA_DIR" \
  --technology-lookup "$LOOKUP" "${EVAL_MASK_ARG[@]}" \
  --num-folds 5 --fold-index "$FOLD" --max-cells 20000 \
  --device cuda --out "$EVAL_OUT"

echo "COMPLETE $TAG fold $FOLD -> $EVAL_OUT"
echo "TASK_WALL_SECONDS=$((SECONDS-task_started))"
