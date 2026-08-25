#!/bin/bash
#SBATCH -J vm_validation
#SBATCH -o vm_validation_%j.out
#SBATCH -e vm_validation_%j.err
#SBATCH -p gh
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 48:00:00
#SBATCH -A MCB26031

# Standalone TACC Slurm validation runner.
# It runs Python modules directly and does not invoke any other shell script.
# Full mode is the default.
# Usage: sbatch run_validation_sbatch.sh [full|test] [--overwrite]
# The default checkpoint is the completed, untagged Gaussian-VAE + CellFlow run.
# Override WORLD_CHECKPOINT and VALIDATION_OUTPUTS with sbatch --export when needed.

# Initialize Conda before enabling strict mode because the user's .bashrc
# contains optional startup commands that are not safe under `set -e`/`set -u`.
source /home1/10119/ghzheng/.bashrc
conda activate /scratch/10119/ghzheng/conda_envs/worldmodeltraining

set -euo pipefail

EXPECTED_PYTHON="/scratch/10119/ghzheng/conda_envs/worldmodeltraining/bin/python"
if [[ "$(command -v python)" != "$EXPECTED_PYTHON" ]]; then
  echo "ERROR: wrong Python executable: $(command -v python || true)"
  echo "Expected: $EXPECTED_PYTHON"
  exit 1
fi

MODE="${1:-full}"
case "$MODE" in
  test|full) ;;
  *) echo "ERROR: mode must be test or full (got: $MODE)"; exit 2 ;;
esac
OVERWRITE="${2:-}"
case "$OVERWRITE" in
  "") overwrite_args=() ;;
  --overwrite) overwrite_args=(--overwrite) ;;
  *) echo "ERROR: optional second argument must be --overwrite"; exit 2 ;;
esac

VM_ROOT="${VM_ROOT:-/scratch/10119/ghzheng/Virtual_microenv}"
VALIDATION_INPUTS="${VALIDATION_INPUTS:-$VM_ROOT/validation_inputs}"
VALIDATION_OUTPUTS="${VALIDATION_OUTPUTS:-$VM_ROOT/validation_outputs}"
WORLD_CHECKPOINT="${WORLD_CHECKPOINT:-$VM_ROOT/checkpoints_1000k_vae_cellflow_drift3_medium_pop_full_data/best_model_level6_fold0.pt}"
PAIRED_RNASEQ_DIR="${PAIRED_RNASEQ_DIR:-$VALIDATION_INPUTS/standardized_rnaseq}"
TCGA_COUNTS_DIR="${TCGA_COUNTS_DIR:-$VALIDATION_INPUTS/standardized_tcga_raw_counts}"
TCGA_CLINICAL_DIR="${TCGA_CLINICAL_DIR:-$VALIDATION_INPUTS/clinical_by_project}"
RANKED_GENE_REFERENCE="${RANKED_GENE_REFERENCE:-$VALIDATION_INPUTS/gene_unification_reference/ranked_gene_symbol_to_ensembl_reference.tsv}"
COMPASS_REPO="${COMPASS_REPO:-$VM_ROOT/external/COMPASS}"
SCGPT_MODEL_DIR="${SCGPT_MODEL_DIR:-$VM_ROOT/external/scgpt_pan_cancer}"
RUN_OUTPUTS="$VALIDATION_OUTPUTS/$MODE"

cd "$VM_ROOT"

# Resolve a Google Drive download that contains one extra directory level.
if [[ ! -f "$SCGPT_MODEL_DIR/best_model.pt" ]]; then
  scgpt_model_file=$(find "$SCGPT_MODEL_DIR" -type f -name best_model.pt -print -quit 2>/dev/null || true)
  if [[ -n "$scgpt_model_file" ]]; then
    SCGPT_MODEL_DIR=$(dirname "$scgpt_model_file")
  fi
fi

required_paths=(
  "$WORLD_CHECKPOINT"
  "$PAIRED_RNASEQ_DIR/patient_pairs.tsv"
  "$TCGA_COUNTS_DIR/expression_files.tsv"
  "$TCGA_CLINICAL_DIR"
  "$RANKED_GENE_REFERENCE"
  "$COMPASS_REPO/baseline/immnue_score/__init__.py"
  "$SCGPT_MODEL_DIR/best_model.pt"
  "$SCGPT_MODEL_DIR/vocab.json"
  "$SCGPT_MODEL_DIR/args.json"
)
for path in "${required_paths[@]}"; do
  if [[ ! -e "$path" ]]; then
    echo "ERROR: required input is missing: $path"
    exit 1
  fi
done

command -v nvidia-smi >/dev/null 2>&1 || {
  echo "ERROR: this RNA validation job requires a GPU compute node."
  exit 1
}

python -c "from RNA_validation.torchtext_compat import install_torchtext_compat; install_torchtext_compat(); from scgpt.tasks import embed_data" >/dev/null
python -c "import gseapy, tidepy, qnorm"

# Fail before expensive stages if validation code and the runner came from
# different versions.
python -m RNA_validation.run_scgpt_benchmark --help | grep -q -- "--raw-input-dir" || {
  echo "ERROR: RNA_validation/run_scgpt_benchmark.py is outdated."
  echo "Upload the complete current RNA_validation directory before resubmitting."
  exit 1
}

if [[ "$MODE" == "test" ]]; then
  folds=2
  epochs=1
  paired_batch_size=8
  tcga_batch_size=8
  ridge_components=4
  bootstrap=50
  scgpt_batch=8
  scgpt_max_length=256
  min_patients=5
  min_events=1
  paired_mode_args=(--limit-datasets 6)
  tcga_mode_args=(--limit-projects 2 --limit-patients-per-project 16)
else
  folds=5
  epochs=30
  paired_batch_size=32
  tcga_batch_size=64
  ridge_components=32
  bootstrap=1000
  scgpt_batch=64
  scgpt_max_length=1200
  min_patients=20
  min_events=5
  paired_mode_args=()
  tcga_mode_args=()
fi

mkdir -p "$RUN_OUTPUTS"

echo "============================================================"
echo "RNA validation"
echo "mode=$MODE"
echo "root=$VM_ROOT"
echo "outputs=$RUN_OUTPUTS"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "============================================================"

echo "[1/12] Prepare paired raw-count RNA-seq"
python -m RNA_validation.prepare_paired_bulk \
  --input-dir "$PAIRED_RNASEQ_DIR" \
  --checkpoint "$WORLD_CHECKPOINT" \
  --ranked-reference "$RANKED_GENE_REFERENCE" \
  --transform auto \
  "${paired_mode_args[@]}" \
  "${overwrite_args[@]}" \
  --output-dir "$RUN_OUTPUTS/paired_prepared"

echo "[2/12] Prepare TCGA raw counts"
python -m RNA_validation.prepare_tcga_bulk \
  --input-dir "$TCGA_COUNTS_DIR" \
  --checkpoint "$WORLD_CHECKPOINT" \
  --ranked-reference "$RANKED_GENE_REFERENCE" \
  --transform auto \
  "${tcga_mode_args[@]}" \
  "${overwrite_args[@]}" \
  --output-dir "$RUN_OUTPUTS/tcga_prepared"

echo "[3/12] Prepare TCGA OS, PFS, and clinical outcomes"
python -m RNA_validation.prepare_tcga_outcomes \
  --tcga-count-dir "$TCGA_COUNTS_DIR" \
  --clinical-root "$TCGA_CLINICAL_DIR" \
  --selected-patients "$RUN_OUTPUTS/tcga_prepared/selected_expression_files.tsv" \
  "${overwrite_args[@]}" \
  --output-dir "$RUN_OUTPUTS/tcga_outcomes"

echo "[4/12] Patient-grouped paired fine-tuning and validation"
python -m RNA_validation.finetune_paired_cv \
  --prepared-dir "$RUN_OUTPUTS/paired_prepared" \
  --checkpoint "$WORLD_CHECKPOINT" \
  --output-dir "$RUN_OUTPUTS/paired_world_model_cv" \
  --group-column biological_patient_id \
  --folds "$folds" \
  --epochs "$epochs" \
  --batch-size "$paired_batch_size" \
  --ridge-components "$ridge_components" \
  --trainable transition \
  "${overwrite_args[@]}" \
  --device cuda

echo "[5/12] Paired COMPASS empirical scores"
python -m RNA_validation.run_empirical_scores \
  --input-kind paired \
  --prepared-dir "$RUN_OUTPUTS/paired_prepared" \
  --prediction-dir "$RUN_OUTPUTS/paired_world_model_cv" \
  --compass-repo "$COMPASS_REPO" \
  --layers pre,observed_post,predicted_post \
  --methods all \
  --bootstrap "$bootstrap" \
  "${overwrite_args[@]}" \
  --output-dir "$RUN_OUTPUTS/paired_empirical_scores"

echo "[6/12] Explain paired predictions with empirical scores"
python -m RNA_validation.analyze_paired_explanations \
  --patient-predictions "$RUN_OUTPUTS/paired_world_model_cv/patient_predictions.tsv" \
  --empirical-scores "$RUN_OUTPUTS/paired_empirical_scores/empirical_scores.tsv.gz" \
  "${overwrite_args[@]}" \
  --output-dir "$RUN_OUTPUTS/paired_score_explanations"

echo "[7/12] scGPT comparison"
python -m RNA_validation.run_scgpt_benchmark \
  --prepared-dir "$RUN_OUTPUTS/paired_prepared" \
  --raw-input-dir "$PAIRED_RNASEQ_DIR" \
  --scgpt-model-dir "$SCGPT_MODEL_DIR" \
  --folds-file "$RUN_OUTPUTS/paired_world_model_cv/patient_predictions.tsv" \
  --output-dir "$RUN_OUTPUTS/paired_scgpt" \
  --batch-size "$scgpt_batch" \
  --max-length "$scgpt_max_length" \
  "${overwrite_args[@]}" \
  --device cuda

echo "[8/12] Fixed-checkpoint TCGA inference"
python -m RNA_validation.infer_tcga \
  --prepared-dir "$RUN_OUTPUTS/tcga_prepared" \
  --checkpoint "$WORLD_CHECKPOINT" \
  --output-dir "$RUN_OUTPUTS/tcga_inference" \
  --batch-size "$tcga_batch_size" \
  "${overwrite_args[@]}" \
  --device cuda

echo "[9/12] TCGA empirical scores"
python -m RNA_validation.run_empirical_scores \
  --input-kind tcga \
  --prepared-dir "$RUN_OUTPUTS/tcga_prepared" \
  --prediction-dir "$RUN_OUTPUTS/tcga_inference" \
  --compass-repo "$COMPASS_REPO" \
  --layers pre,predicted_post \
  --methods all \
  "${overwrite_args[@]}" \
  --output-dir "$RUN_OUTPUTS/tcga_empirical_scores"

echo "[10/12] TCGA OS, PFS, and response stratification"
python -m RNA_validation.analyze_tcga_outcomes \
  --inference-dir "$RUN_OUTPUTS/tcga_inference" \
  --outcomes "$RUN_OUTPUTS/tcga_outcomes/tcga_patient_outcomes.tsv" \
  --empirical-scores "$RUN_OUTPUTS/tcga_empirical_scores/empirical_scores.tsv.gz" \
  --output-dir "$RUN_OUTPUTS/tcga_outcome_analysis" \
  --min-patients "$min_patients" \
  --min-events "$min_events" \
  "${overwrite_args[@]}"

echo "[11/12] Verify expected result files"
for result in \
  "$RUN_OUTPUTS/paired_world_model_cv/patient_predictions.tsv" \
  "$RUN_OUTPUTS/paired_empirical_scores/empirical_scores.tsv.gz" \
  "$RUN_OUTPUTS/paired_scgpt" \
  "$RUN_OUTPUTS/tcga_inference/manifest.tsv" \
  "$RUN_OUTPUTS/tcga_outcome_analysis"
do
  test -e "$result" || { echo "ERROR: expected result is missing: $result"; exit 1; }
done

echo "[12/12] Complete"
echo "Validation finished successfully: $RUN_OUTPUTS"
