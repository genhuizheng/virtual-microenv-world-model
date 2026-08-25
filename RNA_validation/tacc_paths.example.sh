#!/bin/bash
# Copy to RNA_validation/tacc_paths.sh on TACC and edit only if your layout differs.
export VM_ROOT="${VM_ROOT:-/scratch/10119/ghzheng/Virtual_microenv}"
export VALIDATION_INPUTS="${VALIDATION_INPUTS:-${VM_ROOT}/validation_inputs}"
export VALIDATION_OUTPUTS="${VALIDATION_OUTPUTS:-${VM_ROOT}/validation_outputs}"
export VALIDATION_MODE="${VALIDATION_MODE:-full}"
export WORLD_CHECKPOINT="${WORLD_CHECKPOINT:-${VM_ROOT}/checkpoints_1000k_vae_cellflow_drift3_medium_pop_full_data/best_model_level6_fold0.pt}"
export PAIRED_RNASEQ_DIR="${PAIRED_RNASEQ_DIR:-${VALIDATION_INPUTS}/standardized_rnaseq}"
export TCGA_COUNTS_DIR="${TCGA_COUNTS_DIR:-${VALIDATION_INPUTS}/standardized_tcga_raw_counts}"
export TCGA_CLINICAL_DIR="${TCGA_CLINICAL_DIR:-${VALIDATION_INPUTS}/clinical_by_project}"
export RANKED_GENE_REFERENCE="${RANKED_GENE_REFERENCE:-${VALIDATION_INPUTS}/gene_unification_reference/ranked_gene_symbol_to_ensembl_reference.tsv}"
export COMPASS_REPO="${COMPASS_REPO:-${VM_ROOT}/external/COMPASS}"
export SCGPT_MODEL_DIR="${SCGPT_MODEL_DIR:-${VM_ROOT}/external/scgpt_pan_cancer}"
