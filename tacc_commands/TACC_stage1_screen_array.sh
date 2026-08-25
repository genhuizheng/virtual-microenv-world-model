#!/bin/bash
#SBATCH -J vm_s1_screen
#SBATCH -o logs/s1_screen_%A_%a.out
#SBATCH -e logs/s1_screen_%A_%a.err
#SBATCH -p gh
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -a 0-24%4
#SBATCH --mail-user=genhuizheng@utexas.edu
#SBATCH --mail-type=FAIL,END
#SBATCH -t 08:00:00
#SBATCH -A MCB26031

# One Stage-1 configuration per array task, read by index from a TSV.
#
# The TSV's `trainer` column selects which Stage-1 model to train:
#   scvi   -> train_stage1_scvi.py        (scvi-tools; count or Normal likelihood)
#   masked -> train_stage1_masked_vae.py  (own Gaussian VAE; masked reconstruction)
# and `use_mask` selects whether a masked run actually receives the gene masks.
# A masked run with use_mask=no is the architecture-matched control: same
# network, same data, mask off, which is what isolates the effect of masking
# from the effect of changing model family.
#
# Adjust -a to match the config count: `0-N%K` runs N+1 tasks, K concurrently.
# The default 0-24%4 matches the 25 configurations in
# stage1_screen_configs.tsv. Verify with:
#     tail -n +2 stage1_screen_configs.tsv | wc -l
#
# Tasks are independent, so one failure does not take down the screen, and
# completed tasks are skipped on resubmission.
#
# Usage:
#   sbatch tacc_commands/TACC_stage1_screen_array.sh
#   sbatch tacc_commands/TACC_stage1_screen_array.sh stage1_screen_configs.tsv 0 50
#
# Rank once the array finishes (mask-aware scoring is strongly recommended --
# without it, reconstruction credit is partly for predicting structural zeros):
#   python rank_stage1_screen.py --pattern 'eval_screen_*_fold0.json'

set -eo pipefail
task_started=$SECONDS
source /home1/10119/ghzheng/.bashrc
# Override with VM_CONDA_ENV if the environment moves. A bare name is
# deliberately avoided: it only resolves for whichever conda base happens
# to be active, which differs between login and compute nodes.
CONDA_ENV="${VM_CONDA_ENV:-/scratch/10119/ghzheng/conda_envs/worldmodel_withconfidenceot}"
conda activate "$CONDA_ENV" || { echo "ERROR: cannot activate $CONDA_ENV"; exit 1; }
echo "conda_env=$CONDA_ENV"
echo "python=$(command -v python)"

cd "${SLURM_SUBMIT_DIR:-$(dirname "$(readlink -f "$0")")/..}"

CONFIGS="${1:-stage1_screen_configs.tsv}"
FOLD="${2:-0}"
MAX_EPOCHS="${3:-50}"
DATA_DIR="${4:-paired_training_h5ad_50k}"
LOOKUP="${5:-../Virtual_microenv/dataset_platform_lookup.tsv}"
GENE_DIR="${6:-../Virtual_microenv}"
MASK_NPZ="${7:-../Virtual_microenv/gene_masks_by_dataset.npz}"

mkdir -p logs

test -f "$CONFIGS" || { echo "ERROR: missing $CONFIGS"; exit 1; }
test -d "$DATA_DIR" || { echo "ERROR: missing $DATA_DIR"; exit 1; }
test -f "$LOOKUP"  || { echo "ERROR: missing $LOOKUP"; exit 1; }

IDX="${SLURM_ARRAY_TASK_ID:-0}"

row=$(python - "$CONFIGS" "$IDX" <<'PY'
import csv, sys
with open(sys.argv[1], newline="") as handle:
    rows = [r for r in csv.DictReader(handle, delimiter="\t") if r.get("tag")]
idx = int(sys.argv[2])
if idx >= len(rows):
    sys.exit(f"array index {idx} exceeds {len(rows)} configurations")
r = rows[idx]
print("\t".join(r[k] for k in (
    "tag", "trainer", "use_mask", "gene_likelihood", "expression_transform",
    "latent_dim", "n_hidden", "n_layers", "dispersion", "batch_key", "gene_list")))
PY
)
IFS=$'\t' read -r TAG TRAINER USE_MASK LIK XFORM LATENT HIDDEN LAYERS DISP BATCHKEY GENELIST <<< "$row"

case "$GENELIST" in
  90pct)   GENE_ARG=(--keep-gene-list "$GENE_DIR/genes_measured_90pct.txt") ;;
  50pct)   GENE_ARG=(--keep-gene-list "$GENE_DIR/genes_measured_50pct.txt") ;;
  none|"") GENE_ARG=() ;;
  *)       GENE_ARG=(--keep-gene-list "$GENE_DIR/$GENELIST") ;;
esac

CKPT_DIR="stage1_screen_${TAG}"
EVAL_OUT="eval_screen_${TAG}_fold${FOLD}.json"

echo "============================================================"
echo "task=$IDX  tag=$TAG  trainer=$TRAINER  use_mask=$USE_MASK"
echo "  likelihood=$LIK transform=$XFORM latent=$LATENT hidden=$HIDDEN layers=$LAYERS"
echo "  dispersion=$DISP batch_key=$BATCHKEY genes=$GENELIST fold=$FOLD"
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
    ;;
  masked)
    CKPT="$CKPT_DIR/best_stage1_masked_vae_fold${FOLD}.pt"
    MASK_ARG=()
    if [[ "$USE_MASK" == "yes" ]]; then
      test -f "$MASK_NPZ" || { echo "ERROR: missing gene mask npz: $MASK_NPZ"; exit 1; }
      MASK_ARG=(--gene-mask-npz "$MASK_NPZ")
    else
      echo "use_mask=no -> architecture-matched control, mask disabled"
    fi
    python train_stage1_masked_vae.py \
      --data-dir "$DATA_DIR" \
      --checkpoint-dir "$CKPT_DIR" \
      "${MASK_ARG[@]}" \
      --max-epochs "$MAX_EPOCHS" \
      --batch-size 256 \
      --latent-dim "$LATENT" \
      --n-hidden "$HIDDEN" \
      --n-layers "$LAYERS" \
      --expression-transform "$XFORM" \
      --num-folds 5 --fold-index "$FOLD" \
      --group-column patient_id \
      --batch-key "$BATCHKEY" \
      "${GENE_ARG[@]}" \
      --min-cells-detected 1 \
      --technology-lookup "$LOOKUP" \
      --keep-technologies "10x genomics" \
      --device cuda --overwrite
    ;;
  *)
    echo "ERROR: unknown trainer '$TRAINER' for tag '$TAG' (expected scvi or masked)"
    exit 2
    ;;
esac

test -f "$CKPT" || { echo "ERROR: checkpoint not written: $CKPT"; exit 1; }

# Reconstruction is scored only over measured genes, for every configuration.
# A structural zero is not a real target, so scoring against it measures
# agreement with padding -- which silently rewards models that predict zeros.
EVAL_MASK_ARG=()
if [[ -f "$MASK_NPZ" ]]; then
  EVAL_MASK_ARG=(--gene-mask-npz "$MASK_NPZ")
fi

python evaluate_stage1_scvi.py \
  --checkpoint "$CKPT" \
  --data-dir "$DATA_DIR" \
  --technology-lookup "$LOOKUP" \
  "${EVAL_MASK_ARG[@]}" \
  --num-folds 5 --fold-index "$FOLD" \
  --max-cells 20000 \
  --device cuda \
  --out "$EVAL_OUT"

echo "COMPLETE $TAG -> $EVAL_OUT"
echo "TASK_WALL_SECONDS=$((SECONDS-task_started))"
