#!/bin/bash
# Run the complete validation pipeline directly inside an active TACC idev session.
# Usage:
#   bash tacc_commands/run_validation_idev.sh test
#   bash tacc_commands/run_validation_idev.sh full

set -euo pipefail

VALIDATION_MODE="${1:-test}"
case "$VALIDATION_MODE" in
  test|full) ;;
  *) echo "ERROR: mode must be test or full (got: $VALIDATION_MODE)"; exit 2 ;;
esac
export VALIDATION_MODE
# Some TACC .bashrc configurations read this variable without a default.
# Export it for any legacy stage scripts that still source .bashrc under `set -u`.
export SHELL_STARTUP_DEBUG="${SHELL_STARTUP_DEBUG:-0}"

VM_ROOT="${VM_ROOT:-/scratch/10119/ghzheng/Virtual_microenv}"
cd "$VM_ROOT"

if [[ ! -f RNA_validation/tacc_paths.sh ]]; then
  cp RNA_validation/tacc_paths.example.sh RNA_validation/tacc_paths.sh
  echo "Created RNA_validation/tacc_paths.sh from the example defaults."
fi
source RNA_validation/tacc_paths.sh

# The Google Drive download can create one extra directory level. Resolve it.
if [[ ! -f "${SCGPT_MODEL_DIR:-}/best_model.pt" ]]; then
  scgpt_model_file=$(find "$VM_ROOT/external/scgpt_pan_cancer" -type f -name best_model.pt -print -quit 2>/dev/null || true)
  if [[ -n "$scgpt_model_file" ]]; then
    SCGPT_MODEL_DIR=$(dirname "$scgpt_model_file")
    export SCGPT_MODEL_DIR
  fi
fi

echo "============================================================"
echo "Interactive validation"
echo "mode=$VALIDATION_MODE"
echo "root=$VM_ROOT"
echo "scgpt_model_dir=${SCGPT_MODEL_DIR:-missing}"
echo "============================================================"

required_paths=(
  "$WORLD_CHECKPOINT"
  "$PAIRED_RNASEQ_DIR/patient_pairs.tsv"
  "$TCGA_COUNTS_DIR/expression_files.tsv"
  "$TCGA_CLINICAL_DIR"
  "$RANKED_GENE_REFERENCE"
  "$COMPASS_REPO/baseline/immnue_score/__init__.py"
  "${SCGPT_MODEL_DIR:-}/best_model.pt"
  "${SCGPT_MODEL_DIR:-}/vocab.json"
  "${SCGPT_MODEL_DIR:-}/args.json"
)
for path in "${required_paths[@]}"; do
  if [[ ! -e "$path" ]]; then
    echo "ERROR: required input is missing: $path"
    exit 1
  fi
done

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "ERROR: no GPU command found. Run this script inside an active GPU idev session."
  exit 1
fi
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

if ! python -c "from RNA_validation.torchtext_compat import install_torchtext_compat; install_torchtext_compat(); from scgpt.tasks import embed_data" >/dev/null 2>&1; then
  echo "ERROR: the required scGPT embedding import is unavailable in the active environment."
  echo "Install only the package code with: python -m pip install --no-deps scgpt==0.2.4"
  exit 1
fi

if ! python -c "import gseapy, tidepy, qnorm" >/dev/null 2>&1; then
  echo "ERROR: COMPASS baseline dependencies are missing."
  echo "Install them with: python -m pip install gseapy tidepy qnorm"
  exit 1
fi

echo "[1/5] Preparing paired RNA-seq and TCGA inputs"
bash tacc_commands/TACC_validation_01_prepare.sh

echo "[2/5] Running patient-grouped paired fine-tuning and validation"
bash tacc_commands/TACC_validation_02_paired_cv.sh

echo "[3/5] Running paired empirical scores and scGPT comparison"
bash tacc_commands/TACC_validation_03_paired_comparators.sh

echo "[4/5] Running fixed-checkpoint TCGA inference and empirical scores"
bash tacc_commands/TACC_validation_04_tcga_inference.sh

echo "[5/5] Running TCGA OS/PFS/response analysis"
bash tacc_commands/TACC_validation_05_tcga_analysis.sh

echo "============================================================"
echo "VALIDATION COMPLETE"
echo "mode=$VALIDATION_MODE"
echo "outputs=$VALIDATION_OUTPUTS/$VALIDATION_MODE"
echo "============================================================"
