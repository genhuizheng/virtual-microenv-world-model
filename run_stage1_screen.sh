#!/bin/bash
# Screen Stage-1 VAE configurations on one fold, then evaluate each identically.
#
# Mirrors the screening design already used for Stage 2: sample the space on a
# single fold, rank, and promote only the top few to full cross-validation.
# Comparing two hand-picked configurations on one fold cannot separate a real
# difference from fold noise; a ranked screen at least shows which axes matter.
#
# Usage:
#   bash run_stage1_screen.sh <configs.tsv> <fold> <data_dir> <lookup> <gene_dir> [max_epochs]
#
# Example:
#   bash run_stage1_screen.sh stage1_screen_configs.tsv 0 \
#        paired_training_h5ad_50k \
#        ../Virtual_microenv/dataset_platform_lookup.tsv \
#        ../Virtual_microenv 50
#
# gene_dir must contain genes_measured_90pct.txt and genes_measured_50pct.txt.
# Every run writes stage1_screen_<tag>/ and eval_screen_<tag>_fold<N>.json;
# rank with rank_stage1_screen.py.

set -u

CONFIGS="${1:?configs tsv required}"
FOLD="${2:-0}"
DATA_DIR="${3:-paired_training_h5ad_50k}"
LOOKUP="${4:-../Virtual_microenv/dataset_platform_lookup.tsv}"
GENE_DIR="${5:-../Virtual_microenv}"
MAX_EPOCHS="${6:-50}"

test -f "$CONFIGS" || { echo "ERROR: missing $CONFIGS"; exit 1; }
test -d "$DATA_DIR" || { echo "ERROR: missing $DATA_DIR"; exit 1; }
test -f "$LOOKUP" || { echo "ERROR: missing $LOOKUP"; exit 1; }

echo "============================================================"
echo "Stage-1 configuration screen"
echo "configs=$CONFIGS  fold=$FOLD  data=$DATA_DIR  epochs=$MAX_EPOCHS"
echo "============================================================"

N_OK=0
N_FAIL=0
FAILED=""

# skip header; read tab-separated fields
tail -n +2 "$CONFIGS" | while IFS=$'\t' read -r TAG LIK XFORM LATENT HIDDEN LAYERS DISP BATCHKEY GENELIST NOTE; do
  [ -z "${TAG:-}" ] && continue
  case "$TAG" in \#*) continue ;; esac

  case "$GENELIST" in
    90pct) GENE_ARG=(--keep-gene-list "$GENE_DIR/genes_measured_90pct.txt") ;;
    50pct) GENE_ARG=(--keep-gene-list "$GENE_DIR/genes_measured_50pct.txt") ;;
    none|"") GENE_ARG=() ;;
    *) GENE_ARG=(--keep-gene-list "$GENE_DIR/$GENELIST") ;;
  esac

  CKPT_DIR="stage1_screen_${TAG}"
  EVAL_OUT="eval_screen_${TAG}_fold${FOLD}.json"

  echo ""
  echo "------------------------------------------------------------"
  echo "[$TAG] $NOTE"
  echo "  likelihood=$LIK transform=$XFORM latent=$LATENT hidden=$HIDDEN layers=$LAYERS"
  echo "  dispersion=$DISP batch_key=$BATCHKEY genes=$GENELIST"
  echo "------------------------------------------------------------"

  if ! python train_stage1_scvi.py \
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
      --device cuda --overwrite; then
    echo "[$TAG] TRAIN FAILED -- continuing"
    continue
  fi

  if ! python evaluate_stage1_scvi.py \
      --checkpoint "$CKPT_DIR/best_stage1_scvi_fold${FOLD}.pt" \
      --data-dir "$DATA_DIR" \
      --technology-lookup "$LOOKUP" \
      --num-folds 5 --fold-index "$FOLD" \
      --max-cells 20000 \
      --device cuda \
      --out "$EVAL_OUT"; then
    echo "[$TAG] EVAL FAILED -- continuing"
    continue
  fi

  echo "[$TAG] done -> $EVAL_OUT"
done

echo ""
echo "============================================================"
echo "Screen finished. Rank with:"
echo "  python rank_stage1_screen.py --pattern 'eval_screen_*_fold${FOLD}.json'"
echo "============================================================"
