#!/bin/bash
# Serial Stage-1 configuration screen. Prefer the SLURM array
# (tacc_commands/TACC_stage1_screen_array.sh) on a cluster -- it runs several
# configs concurrently, isolates failures, and resumes. This exists for
# interactive runs, small grids, and debugging.
#
# The TSV's `trainer` column selects the Stage-1 model:
#   scvi   -> train_stage1_scvi.py        (scvi-tools; count or Normal likelihood)
#   masked -> train_stage1_masked_vae.py  (own Gaussian VAE; masked reconstruction)
# and `use_mask` decides whether a masked run receives the gene masks. A masked
# run with use_mask=no is the architecture-matched control: same network, mask
# off, which is what separates the effect of masking from the effect of
# switching model family.
#
# Usage:
#   bash run_stage1_screen.sh <configs.tsv> <fold> <data_dir> <lookup> <gene_dir> \
#                             [epochs] [mask_npz] [device] [max_eval_cells]
#
# Example:
#   bash run_stage1_screen.sh stage1_screen_configs.tsv 0 \
#        paired_training_h5ad_50k \
#        ../Virtual_microenv/dataset_platform_lookup.tsv \
#        ../Virtual_microenv 50 \
#        ../Virtual_microenv/gene_masks_by_dataset.npz
#
# gene_dir must contain genes_measured_90pct.txt and genes_measured_50pct.txt.
# Rank with rank_stage1_screen.py once finished.

set -u

CONFIGS="${1:?configs tsv required}"
FOLD="${2:-0}"
DATA_DIR="${3:-paired_training_h5ad_50k}"
LOOKUP="${4:-../Virtual_microenv/dataset_platform_lookup.tsv}"
GENE_DIR="${5:-../Virtual_microenv}"
MAX_EPOCHS="${6:-50}"
MASK_NPZ="${7:-../Virtual_microenv/gene_masks_by_dataset.npz}"
DEVICE="${8:-cuda}"
MAX_EVAL_CELLS="${9:-20000}"

test -f "$CONFIGS" || { echo "ERROR: missing $CONFIGS"; exit 1; }
test -d "$DATA_DIR" || { echo "ERROR: missing $DATA_DIR"; exit 1; }
test -f "$LOOKUP" || { echo "ERROR: missing $LOOKUP"; exit 1; }

echo "============================================================"
echo "Stage-1 configuration screen (serial)"
echo "configs=$CONFIGS  fold=$FOLD  data=$DATA_DIR  epochs=$MAX_EPOCHS"
echo "mask_npz=$MASK_NPZ  device=$DEVICE"
echo "============================================================"

tail -n +2 "$CONFIGS" | while IFS=$'\t' read -r TAG TRAINER USE_MASK LIK XFORM LATENT HIDDEN LAYERS DISP BATCHKEY GENELIST NOTE; do
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
  echo "  trainer=$TRAINER use_mask=$USE_MASK likelihood=$LIK transform=$XFORM"
  echo "  latent=$LATENT hidden=$HIDDEN layers=$LAYERS batch_key=$BATCHKEY genes=$GENELIST"
  echo "------------------------------------------------------------"

  if [ -f "$EVAL_OUT" ]; then
    echo "[$TAG] already complete -> $EVAL_OUT"
    continue
  fi

  if [ "$TRAINER" = "masked" ]; then
    CKPT="$CKPT_DIR/best_stage1_masked_vae_fold${FOLD}.pt"
    MASK_ARG=()
    if [ "$USE_MASK" = "yes" ]; then
      if [ ! -f "$MASK_NPZ" ]; then
        echo "[$TAG] ERROR: missing gene mask npz: $MASK_NPZ -- skipping"
        continue
      fi
      MASK_ARG=(--gene-mask-npz "$MASK_NPZ")
    else
      echo "[$TAG] use_mask=no -> architecture-matched control, mask disabled"
    fi
    if ! python train_stage1_masked_vae.py \
        --data-dir "$DATA_DIR" --checkpoint-dir "$CKPT_DIR" "${MASK_ARG[@]}" \
        --max-epochs "$MAX_EPOCHS" --batch-size 256 \
        --latent-dim "$LATENT" --n-hidden "$HIDDEN" --n-layers "$LAYERS" \
        --expression-transform "$XFORM" \
        --num-folds 5 --fold-index "$FOLD" --group-column patient_id \
        --batch-key "$BATCHKEY" "${GENE_ARG[@]}" --min-cells-detected 1 \
        --technology-lookup "$LOOKUP" --keep-technologies "10x genomics" \
        --device "$DEVICE" --overwrite; then
      echo "[$TAG] TRAIN FAILED -- continuing"
      continue
    fi
  else
    CKPT="$CKPT_DIR/best_stage1_scvi_fold${FOLD}.pt"
    if ! python train_stage1_scvi.py \
        --data-dir "$DATA_DIR" --checkpoint-dir "$CKPT_DIR" \
        --max-epochs "$MAX_EPOCHS" --batch-size 256 \
        --latent-dim "$LATENT" --n-hidden "$HIDDEN" --n-layers "$LAYERS" \
        --dispersion "$DISP" --gene-likelihood "$LIK" \
        --expression-transform "$XFORM" \
        --num-folds 5 --fold-index "$FOLD" --group-column patient_id \
        --batch-key "$BATCHKEY" "${GENE_ARG[@]}" --min-cells-detected 1 \
        --technology-lookup "$LOOKUP" --keep-technologies "10x genomics" \
        --device "$DEVICE" --overwrite; then
      echo "[$TAG] TRAIN FAILED -- continuing"
      continue
    fi
  fi

  # Score reconstruction only over measured genes, for every configuration --
  # a structural zero is not a real target, and scoring against it silently
  # rewards models for predicting padding.
  EVAL_MASK_ARG=()
  if [ -f "$MASK_NPZ" ]; then
    EVAL_MASK_ARG=(--gene-mask-npz "$MASK_NPZ")
  fi

  if ! python evaluate_stage1_scvi.py \
      --checkpoint "$CKPT" --data-dir "$DATA_DIR" \
      --technology-lookup "$LOOKUP" "${EVAL_MASK_ARG[@]}" \
      --num-folds 5 --fold-index "$FOLD" --max-cells "$MAX_EVAL_CELLS" \
      --device "$DEVICE" --out "$EVAL_OUT"; then
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
