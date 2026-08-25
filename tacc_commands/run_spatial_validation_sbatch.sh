#!/bin/bash
#SBATCH -J spatial_predict
#SBATCH -o spatial_predict_%j.out
#SBATCH -e spatial_predict_%j.err
#SBATCH -p gh
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 48:00:00
#SBATCH -A MCB26031
#SBATCH --mail-user=genhuizheng@utexas.edu
#SBATCH --mail-type=all

source /home1/10119/ghzheng/.bashrc
conda activate /scratch/10119/ghzheng/conda_envs/worldmodeltraining
cd /scratch/10119/ghzheng/Virtual_microenv

MODE="${1:-test}"
OVERWRITE="${2:-}"
SPATIAL_ROOT="/scratch/10119/ghzheng/spatial_data_storage/GEO_downloads"
CHECKPOINT="${WORLD_CHECKPOINT:-/scratch/10119/ghzheng/Virtual_microenv/checkpoints_1000k_vae_cellflow_drift3_medium_pop_full_data/best_model_level6_fold0.pt}"
COMPASS_REPO="/scratch/10119/ghzheng/Virtual_microenv/external/COMPASS"
REACTOME_GMT="${REACTOME_GMT:-/scratch/10119/ghzheng/Virtual_microenv/external/reactome/ReactomePathways.gmt}"
OUTPUT_ROOT="/scratch/10119/ghzheng/Virtual_microenv/spatial_validation_outputs/${MODE}"

case "$MODE" in
  test)
    MAX_SAMPLES=-1
    MAX_PRE=256
    BATCH_SIZE=16
    GSEA_PERMUTATIONS=100
    DATASET_ARGS=(--datasets GSE240138 GSE288758 GSE322640)
    ;;
  full)
    MAX_SAMPLES=-1
    MAX_PRE=-1
    BATCH_SIZE=32
    GSEA_PERMUTATIONS=1000
    DATASET_ARGS=()
    ;;
  *)
    echo "ERROR: mode must be test or full"
    exit 2
    ;;
esac

OVERWRITE_ARG=()
if [[ "$OVERWRITE" == "--overwrite" ]]; then
  OVERWRITE_ARG=(--overwrite)
fi

test -d "$SPATIAL_ROOT" || { echo "ERROR: missing spatial root: $SPATIAL_ROOT"; exit 1; }
test -f "$CHECKPOINT" || { echo "ERROR: missing checkpoint: $CHECKPOINT"; exit 1; }
test -d "$COMPASS_REPO/baseline/immnue_score" || { echo "ERROR: missing COMPASS immune-score code: $COMPASS_REPO"; exit 1; }
test -f "$REACTOME_GMT" || { echo "ERROR: missing Reactome GMT: $REACTOME_GMT"; exit 1; }
test -f spatial_validation/build_manifest.py || { echo "ERROR: upload spatial_validation"; exit 1; }
python -m spatial_validation.predict_pre_to_post --help >/dev/null
python -m spatial_validation.pathway_enrichment --help >/dev/null

echo "[1/5] Build strict >10,000-gene pretreatment-only manifest"
python -m spatial_validation.build_manifest \
  --spatial-root "$SPATIAL_ROOT" \
  --output-dir "$OUTPUT_ROOT/manifest" \
  --min-genes 10000 \
  --max-samples "$MAX_SAMPLES" \
  "${DATASET_ARGS[@]}" \
  "${OVERWRITE_ARG[@]}"

echo "[2/5] Predict post-treatment biology from pretreatment only"
python -m spatial_validation.predict_pre_to_post \
  --manifest-dir "$OUTPUT_ROOT/manifest" \
  --checkpoint "$CHECKPOINT" \
  --output-dir "$OUTPUT_ROOT/predictions" \
  --batch-size "$BATCH_SIZE" \
  --flow-steps 8 \
  --max-pre-observations "$MAX_PRE" \
  --device cuda \
  "${OVERWRITE_ARG[@]}"

echo "[3/5] Score immune and treatment-response biology"
python -m spatial_validation.score_biology \
  --prediction-dir "$OUTPUT_ROOT/predictions" \
  --compass-repo "$COMPASS_REPO" \
  --output-dir "$OUTPUT_ROOT/biology_scores" \
  "${OVERWRITE_ARG[@]}"

echo "[4/5] Run Reactome enrichment globally and by dataset, response, and segment"
python -m spatial_validation.pathway_enrichment \
  --prediction-dir "$OUTPUT_ROOT/predictions" \
  --reactome-gmt "$REACTOME_GMT" \
  --output-dir "$OUTPUT_ROOT/reactome" \
  --permutations "$GSEA_PERMUTATIONS" \
  "${OVERWRITE_ARG[@]}"

echo "[5/5] Analyze genes, signatures, compartments, spatial hotspots, and response strata"
python -m spatial_validation.analyze_predictions \
  --prediction-dir "$OUTPUT_ROOT/predictions" \
  --biology-score-dir "$OUTPUT_ROOT/biology_scores" \
  --output-dir "$OUTPUT_ROOT/analysis" \
  "${OVERWRITE_ARG[@]}"

echo "Finished: $OUTPUT_ROOT"
