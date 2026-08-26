#!/bin/bash
#SBATCH -J vm_s1_500k
#SBATCH -o logs/s1_500k_%j.out
#SBATCH -e logs/s1_500k_%j.err
#SBATCH -p gh
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --mail-user=genhuizheng@utexas.edu
#SBATCH --mail-type=FAIL,END
#SBATCH -t 12:00:00
#SBATCH -A MCB26031

# Refit the CHOSEN Stage-1 representation (p90_nob_nb) on the 500k pair set.
#
# This is deliberately a single job, not a screen. The 25-config screen at 50k
# answered *which configuration* -- panel, likelihood, batch key, transform --
# and those margins came from effects that do not depend on data volume:
# binning was catastrophic (Pearson 0.69 -> 0.41), the masked VAE was
# degenerate (role ~1.0 across every config), the likelihood ladder was flat
# within 0.5%, and 15k genes ~= 26k genes. Ten times the data does not reverse
# a 40% gap or create separation in a ladder that was already flat, so only the
# winner's *weights* need refitting.
#
# Why refitting is REQUIRED rather than optional: compute_group_to_fold used to
# assign folds by position in a shuffled roster, so the 50k directory (406
# patients) and the 500k one (408) disagreed on 83.5% of assignments. A Stage-1
# encoder trained on 50k fold-0-train had therefore already seen 68 of the ~82
# patients in 500k fold-0-val. That is fixed now (hash-based assignment in
# sample_paired_h5ad_dataloader.py), but the fix also *changes* fold membership
# relative to every pre-fix checkpoint -- so any checkpoint trained before it
# must not be reused. See docs/HANDOFF_20260826.md.
#
# Config (from stage1_screen_configs.tsv, row p90_nob_nb):
#   scVI | NB likelihood | raw counts | latent 128 | hidden 512 | layers 2
#   dispersion gene | NO batch key | 90pct panel (15,163 genes) | 10x only
#
# Note the gene panel may come out slightly different at 500k: the
# --min-cells-detected filter is recomputed on the training split only. That is
# harmless -- Stage 2 adopts gene_ids from the checkpoint -- but it does mean
# n_genes will not necessarily equal the 50k run's.
#
# Usage:
#   sbatch tacc_commands/TACC_stage1_500k.sh
#   sbatch tacc_commands/TACC_stage1_500k.sh paired_training_h5ad_500k_fast 0 15

set -eo pipefail
job_started=$SECONDS
if [[ -z "${VM_SKIP_ENV:-}" ]]; then
  [[ -f /home1/10119/ghzheng/.bashrc ]] && source /home1/10119/ghzheng/.bashrc || true
  CONDA_ENV="${VM_CONDA_ENV:-/scratch/10119/ghzheng/conda_envs/worldmodel_withconfidenceot}"
  conda activate "$CONDA_ENV" || { echo "ERROR: cannot activate $CONDA_ENV"; exit 1; }
  echo "conda_env=$CONDA_ENV"
fi
echo "python=$(command -v python)"

cd "${SLURM_SUBMIT_DIR:-$(dirname "$(readlink -f "$0")")/..}"

DATA_DIR="${1:-paired_training_h5ad_500k_fast}"
FOLD="${2:-0}"
# 500k pairs -> ~1560 train steps/epoch at batch 256, vs ~157 at 50k. The 50k
# screen used 50 epochs (7,850 steps); 15 epochs here is ~23,000 steps, so this
# is MORE optimisation, not less, despite the smaller epoch count.
MAX_EPOCHS="${3:-15}"
GENE_DIR="${4:-../Virtual_microenv}"
LOOKUP="${5:-$GENE_DIR/dataset_platform_lookup.tsv}"
MASK_NPZ="${6:-$GENE_DIR/gene_masks_by_dataset.npz}"

CKPT_DIR="stage1_500k_p90_nob_nb"
EVAL_OUT="eval_stage1_500k_p90_nob_nb_fold${FOLD}.json"

mkdir -p logs
test -d "$DATA_DIR" || { echo "ERROR: missing $DATA_DIR"; exit 1; }
test -f "$LOOKUP"   || { echo "ERROR: missing $LOOKUP"; exit 1; }
test -f "$GENE_DIR/genes_measured_90pct.txt" || { echo "ERROR: missing gene panel"; exit 1; }

echo "============================================================"
echo "Stage-1 refit on 500k  |  config=p90_nob_nb  fold=$FOLD  epochs=$MAX_EPOCHS"
echo "  data=$DATA_DIR"
echo "  out=$CKPT_DIR"
echo "============================================================"

python train_stage1_scvi.py \
  --data-dir "$DATA_DIR" \
  --checkpoint-dir "$CKPT_DIR" \
  --max-epochs "$MAX_EPOCHS" \
  --batch-size 256 \
  --latent-dim 128 \
  --n-hidden 512 \
  --n-layers 2 \
  --dispersion gene \
  --gene-likelihood nb \
  --expression-transform none \
  --num-folds 5 --fold-index "$FOLD" \
  --group-column patient_id \
  --batch-key none \
  --keep-gene-list "$GENE_DIR/genes_measured_90pct.txt" \
  --min-cells-detected 1 \
  --technology-lookup "$LOOKUP" \
  --keep-technologies "10x genomics" \
  --device cuda --overwrite

CKPT="$CKPT_DIR/best_stage1_scvi_fold${FOLD}.pt"
test -f "$CKPT" || { echo "ERROR: checkpoint not written: $CKPT"; exit 1; }

# Mask-aware scoring: a structural zero in the zero-filled panel is padding,
# not a measured value, so scoring against it rewards predicting padding.
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

echo "COMPLETE -> $CKPT"
echo "eval     -> $EVAL_OUT"
echo "JOB_WALL_SECONDS=$((SECONDS-job_started))"
