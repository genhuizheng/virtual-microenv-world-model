# Bulk RNA validation on TACC

## Confirmed current model run

`TACC_1000k_full_cellflow_level6.sh` is the full one-million-pair Level-6 CellFlow run.
It uses all paired shards (`num_folds=1`) for 50 epochs. The expected checkpoint is:

```text
checkpoints_1000k_full_cellflow_level6_full_data/best_model_level6_fold0.pt
```

`fold0` in that filename is a full-data artifact, not a held-out fold.

## Upload layout

Use this layout under `/scratch/10119/ghzheng/Virtual_microenv`:

```text
Virtual_microenv/
  RNA_validation/
  tacc_commands/
  five_level_cell_world_model.py
  checkpoints_1000k_full_cellflow_level6_full_data/
    best_model_level6_fold0.pt
  validation_inputs/
    standardized_rnaseq/
      studies.tsv
      patients.tsv
      samples.tsv
      patient_pairs.tsv
      counts/
    standardized_tcga_raw_counts/
      patients.tsv
      samples.tsv
      expression_files.tsv
      counts/
    clinical_by_project/
      TCGA-ACC/TCGA-ACC_clinical_raw.json
      ... one raw JSON directory per TCGA project ...
    gene_unification_reference/
      ranked_gene_symbol_to_ensembl_reference.tsv
  external/
    COMPASS/
    scgpt_pan_cancer/
```

The validation programs never modify these input directories. Generated data go to:

```text
/scratch/10119/ghzheng/Virtual_microenv/validation_outputs/test
/scratch/10119/ghzheng/Virtual_microenv/validation_outputs/full
```

Copy `RNA_validation/tacc_paths.example.sh` to `RNA_validation/tacc_paths.sh` on TACC. The
defaults already use the layout above.

## Required local uploads

Upload the first dataset directory:

```text
F:\Virtual_microenv\RNA\standardized_rnaseq
```

to:

```text
validation_inputs/standardized_rnaseq
```

Upload the second count package:

```text
F:\RNA_pretreatment\standardized_tcga_raw_counts
```

to:

```text
validation_inputs/standardized_tcga_raw_counts
```

For TCGA outcomes, only the 33 `TCGA-*/*_clinical_raw.json` files are required from:

```text
F:\RNA_pretreatment\clinical_by_project
```

The large `*_clinical_full.tsv` files are not required.

The ranked mapping file is created by the existing gene-unification workflow:

```text
gene_unification_reference/ranked_gene_symbol_to_ensembl_reference.tsv
```

The existing code under `F:\Virtual_microenv\Code\gene_unification` is treated as a
read-only source. This validation suite does not edit or require uploading that whole directory;
upload only its generated ranked-reference TSV shown above.

## Test and full modes

The same five-job pipeline supports two explicit modes, with separate output trees.

`test` is an end-to-end smoke run using the checkpoint on a deterministic subset:

- first 6 sorted paired-RNA datasets;
- first 2 sorted TCGA projects, at most 16 patients per project;
- 2 patient-grouped folds and 1 fine-tuning epoch;
- 50 empirical-score bootstrap replicates;
- shorter scGPT sequences and smaller GPU batches;
- relaxed TCGA analysis thresholds suitable only for checking execution.

Run it first:

```bash
VALIDATION_MODE=test bash tacc_commands/submit_validation_pipeline.sh
```

Successful test outputs appear under `validation_outputs/test`. These results are a pipeline
check, not reportable performance estimates.

`full` uses all selected records, 5 patient-grouped folds, 30 epochs, 1,000 bootstrap
replicates, and the normal TCGA analysis thresholds:

```bash
VALIDATION_MODE=full bash tacc_commands/submit_validation_pipeline.sh
```

Successful full outputs appear under `validation_outputs/full`. If `VALIDATION_MODE` is omitted,
the scripts default to `full`. Each output directory is intentionally non-overwriting; move or
remove a previous failed mode directory before rerunning that same mode.

## Gene-order contract

The checkpoint's `gene_ids` list is authoritative. Preparation performs:

1. versionless Ensembl matching;
2. ranked gene-symbol to Ensembl matching when exact IDs are unavailable;
3. summation when multiple source rows map to one model gene;
4. zero filling for model genes absent from the bulk cohort;
5. exact reordering to checkpoint order.

Every preparation run writes `model_genes.tsv`, `gene_alignment_summary.tsv`, and a
manifest containing the expression transformation. Stop the run if overlap is unexpectedly low.

## Expression scale

The training-pipeline reference under `F:\Virtual_microenv\Code` confirms that the paired H5AD
shards retain raw counts and that the training dataloader passes `adata.X` and
`adata.layers["target"]` directly to the model. It performs no CPM or log transformation.

The supplied validation jobs therefore use `--transform none`. Model input, fine-tuning targets,
decoded predictions, comparison models, expression metrics, and empirical-score inputs are kept
on their raw-count scale. No CPM, `log1p`, clipping, or composition rescaling is applied.

Raw-count MSE is strongly affected by sequencing depth, and the checkpoint was trained on
single-cell rather than bulk libraries. Retain the per-patient Pearson and response/survival
results alongside MSE, and describe this limitation when interpreting the external validation.

## Dataset 1 analysis

The paired workflow:

- groups folds by `biological_patient_id`, not dataset;
- keeps every record from one biological patient in one fold;
- fine-tunes the transition component only by default;
- predicts held-out post-treatment expression;
- compares against the pretrained model, identity, mean-delta, and PCA-ridge baselines;
- trains response classifiers only inside each training fold;
- compares scGPT as a pretreatment embedding classifier;
- computes all public COMPASS empirical scores on pre, observed post, and cross-validated predicted post expression.
- ranks empirical scores that clarify each out-of-fold prediction using response AUROC/AUPRC,
  responder effect size, correlation with model probability, and correct-versus-incorrect behavior.

The empirical-score response benchmark defaults to ICI-treated patients. Use
`--response-subset all` only for an explicitly exploratory, treatment-heterogeneous analysis.

## Dataset 2 analysis

The TCGA workflow selects one non-FFPE primary tumor profile per patient. It runs the fixed
checkpoint without comparison models and associates predicted transition features and empirical
scores with:

- OS;
- candidate PFS;
- CR/PR versus SD/PD treatment outcomes.

OS is the strongest endpoint. PFS and response retain review flags because raw GDC records are
heterogeneous. TCGA inference is exploratory because the current world model has no treatment
or action token and therefore predicts a generic learned transition, not a treatment-specific
causal outcome.

## Dependencies

Install the common dependencies into `worldmodeltraining`:

```bash
pip install -r RNA_validation/requirements-validation.txt
```

Install scGPT separately and place its pan-cancer or whole-human checkpoint directory at the
configured `SCGPT_MODEL_DIR`. Clone the official COMPASS repository into `external/COMPASS`.

## Submit

After preparation inputs and external tools are present, copy the path configuration once:

```bash
cp RNA_validation/tacc_paths.example.sh RNA_validation/tacc_paths.sh
```

Then use the `test` command above, inspect its job logs and manifests, and submit `full`. The
submission helper applies dependencies so downstream jobs start only after their required
preparation or inference job succeeds.

## Run directly inside idev

When a GPU `idev` allocation is already active, do not use the Slurm submission helper. Run all
five stages sequentially in the current allocation:

```bash
cd /scratch/10119/ghzheng/Virtual_microenv
python -m pip install --no-deps scgpt==0.2.4
python -m pip install gseapy tidepy qnorm
python -c "from scgpt.tasks import embed_data; print('scGPT embedding import OK')"
bash tacc_commands/run_validation_idev.sh test
```

After the test completes successfully, start a sufficiently long `idev` allocation and run:

```bash
bash tacc_commands/run_validation_idev.sh full
```

The interactive runner checks the GPU, required inputs, COMPASS dependencies, scGPT package and
checkpoint files before preparation begins. It also detects an extra directory level created by
the Google Drive folder download.

TACC currently provides a newer ARM64 PyTorch than the final compatible TorchText release. Do not
install TorchText into `worldmodeltraining`. The scGPT benchmark installs a small pure-Python
vocabulary compatibility layer before importing scGPT; this replaces only the TorchText vocabulary
API used by scGPT's tokenizer and leaves PyTorch unchanged.
