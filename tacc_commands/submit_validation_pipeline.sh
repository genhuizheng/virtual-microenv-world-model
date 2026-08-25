#!/bin/bash
set -euo pipefail

VALIDATION_MODE="${VALIDATION_MODE:-full}"
case "$VALIDATION_MODE" in
  test|full) ;;
  *) echo "ERROR VALIDATION_MODE must be test or full (got: $VALIDATION_MODE)"; exit 2 ;;
esac
export VALIDATION_MODE
echo "Submitting validation pipeline in mode=$VALIDATION_MODE"

prepare_job=$(sbatch --parsable --export="ALL,VALIDATION_MODE=${VALIDATION_MODE}" tacc_commands/TACC_validation_01_prepare.sh)
paired_job=$(sbatch --parsable --export="ALL,VALIDATION_MODE=${VALIDATION_MODE}" --dependency="afterok:${prepare_job}" tacc_commands/TACC_validation_02_paired_cv.sh)
paired_compare_job=$(sbatch --parsable --export="ALL,VALIDATION_MODE=${VALIDATION_MODE}" --dependency="afterok:${paired_job}" tacc_commands/TACC_validation_03_paired_comparators.sh)
tcga_infer_job=$(sbatch --parsable --export="ALL,VALIDATION_MODE=${VALIDATION_MODE}" --dependency="afterok:${prepare_job}" tacc_commands/TACC_validation_04_tcga_inference.sh)
tcga_analysis_job=$(sbatch --parsable --export="ALL,VALIDATION_MODE=${VALIDATION_MODE}" --dependency="afterok:${tcga_infer_job}" tacc_commands/TACC_validation_05_tcga_analysis.sh)

echo "prepare_job=${prepare_job}"
echo "paired_cv_job=${paired_job}"
echo "paired_comparators_job=${paired_compare_job}"
echo "tcga_inference_job=${tcga_infer_job}"
echo "tcga_analysis_job=${tcga_analysis_job}"
