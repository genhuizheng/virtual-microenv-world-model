#!/bin/bash
#SBATCH -J vm_posthoc_analysis
#SBATCH -o vm_posthoc_analysis_%j.out
#SBATCH -e vm_posthoc_analysis_%j.err
#SBATCH -p gg
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 04:00:00
#SBATCH -A MCB26031

# Run post-hoc fine-tuning comparison and repeat cancer-specific TCGA analysis.
# Usage: sbatch run_tcga_analysis_sbatch.sh test
#        sbatch run_tcga_analysis_sbatch.sh full

source /home1/10119/ghzheng/.bashrc
conda activate /scratch/10119/ghzheng/conda_envs/worldmodeltraining

set -euo pipefail

EXPECTED_PYTHON="/scratch/10119/ghzheng/conda_envs/worldmodeltraining/bin/python"
if [[ "$(command -v python)" != "$EXPECTED_PYTHON" ]]; then
  echo "ERROR: wrong Python executable: $(command -v python || true)"
  echo "Expected: $EXPECTED_PYTHON"
  exit 1
fi

MODE="${1:-test}"
case "$MODE" in
  test) min_patients=5; min_events=1 ;;
  full) min_patients=20; min_events=5 ;;
  *) echo "ERROR: mode must be test or full (got: $MODE)"; exit 2 ;;
esac

VM_ROOT="${VM_ROOT:-/scratch/10119/ghzheng/Virtual_microenv}"
VALIDATION_OUTPUTS="${VALIDATION_OUTPUTS:-$VM_ROOT/validation_outputs}"
RUN_OUTPUTS="$VALIDATION_OUTPUTS/$MODE"
cd "$VM_ROOT"

for required in \
  "$RUN_OUTPUTS/paired_prepared/patients.tsv" \
  "$RUN_OUTPUTS/paired_world_model_cv/expression_metrics_by_patient.tsv" \
  "$RUN_OUTPUTS/tcga_inference/manifest.tsv" \
  "$RUN_OUTPUTS/tcga_outcomes/tcga_patient_outcomes.tsv" \
  "$RUN_OUTPUTS/tcga_empirical_scores/empirical_scores.tsv.gz"
do
  test -e "$required" || { echo "ERROR: missing completed-stage result: $required"; exit 1; }
done

echo "Compare pretrained versus fine-tuned model on identical held-out patients"
python -m RNA_validation.compare_finetuning \
  --cv-dir "$RUN_OUTPUTS/paired_world_model_cv" \
  --prepared-dir "$RUN_OUTPUTS/paired_prepared"

echo "Repeat cancer-specific TCGA OS/PFS/response analysis"
python -m RNA_validation.analyze_tcga_outcomes \
  --inference-dir "$RUN_OUTPUTS/tcga_inference" \
  --outcomes "$RUN_OUTPUTS/tcga_outcomes/tcga_patient_outcomes.tsv" \
  --empirical-scores "$RUN_OUTPUTS/tcga_empirical_scores/empirical_scores.tsv.gz" \
  --output-dir "$RUN_OUTPUTS/tcga_outcome_analysis" \
  --min-patients "$min_patients" \
  --min-events "$min_events" \
  --overwrite

test -e "$RUN_OUTPUTS/tcga_outcome_analysis/run_summary.json" || {
  echo "ERROR: TCGA analysis summary was not created"
  exit 1
}
test -e "$RUN_OUTPUTS/paired_world_model_cv/finetuning_comparison_summary.tsv" || {
  echo "ERROR: fine-tuning comparison summary was not created"
  exit 1
}

echo "Post-hoc analyses completed under: $RUN_OUTPUTS"
