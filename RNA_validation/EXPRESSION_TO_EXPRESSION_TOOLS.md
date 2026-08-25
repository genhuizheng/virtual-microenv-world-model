# Gene-expression prediction tools and validation plan

## Objective

The paired dataset contains pretreatment and post-treatment bulk RNA-seq from the same biological patients. The expression-prediction task is:

```text
pretreatment gene expression -> predicted post-treatment gene expression
```

An eligible expression comparator must return a vector over genes on a declared expression scale. A model that returns only an embedding or a clinical-response probability cannot be compared with post-treatment expression using MSE.

## Tool summary

| Tool | Principal input | Principal output | Role in this project |
|---|---|---|---|
| World model | Pretreatment expression | Predicted post-treatment expression | Primary model |
| Identity | Pretreatment expression | Pretreatment expression as the post prediction | Required null baseline |
| Mean delta | Pretreatment expression plus training-patient mean change | Predicted post-treatment expression | Required simple treatment baseline |
| PCA-ridge | Pretreatment expression and paired training data | Predicted post-treatment expression | Required supervised baseline; implemented |
| Reduced-rank regression | Pretreatment expression and paired training data | Predicted post-treatment expression | Recommended additional baseline |
| [scGen](https://github.com/theislab/scgen) | Control expression plus condition | Perturbed expression | Recommended external expression model; originally designed for single-cell data |
| [CellOT](https://github.com/bunnech/cellot) | Untreated expression population and treated training population | Transported/predicted treated expression population | Optional distribution-level comparator; originally designed for unpaired single-cell populations |
| [CPA](https://github.com/theislab/cpa) | Expression plus perturbation, dose and context | Perturbed expression | Consider only if treatment metadata and sample size are sufficient |
| [PRnet](https://github.com/Perturbation-Response-Prediction/PRnet) | Baseline expression plus chemical representation | Perturbed expression | Supports bulk expression, but is oriented toward chemical/small-molecule perturbations rather than antibody ICI regimens |
| scGPT `embed_data` | Expression mapped through the checkpoint vocabulary | Patient/sample embedding | Response-classification baseline only |
| scGPT perturbation model | Expression plus explicit perturbation and task-specific fine-tuning | Perturbed expression | Separate training project; not provided by `embed_data` |
| [GEARS](https://github.com/snap-stanford/GEARS) | Expression plus genetic perturbation | Expression after CRISPR/gene perturbation | Not appropriate for bulk ICI treatment validation |

## CellOT interpretation

CellOT learns an optimal-transport map between untreated and treated expression distributions:

```text
training:   untreated population + treated population -> learned transport map
inference:  untreated expression -> predicted treated expression
```

Its output is gene expression, not a responder probability. A separate response classifier would be required to convert CellOT-derived features into a clinical-response probability. CellOT does not require paired control and treated cells; consequently, it does not naturally exploit this dataset's known patient pairing.

## scGPT interpretation

The current scGPT comparator performs:

```text
pretreatment raw counts
  -> independent gene-symbol intersection with scGPT vocab.json
  -> scGPT embedding
  -> patient-grouped logistic classifier
  -> response probability
```

Dataset gene order does not determine scGPT token identity. Gene symbols are mapped to vocabulary IDs. The implementation reads original pretreatment raw counts independently of the world-model gene list, sums duplicate gene-symbol rows and records mapping coverage.

scGPT is not included in post-treatment MSE because `embed_data` returns embeddings rather than post-treatment expression.

## Paired-dataset evaluation

Splits must be grouped by biological patient. For every fold, all model fitting, expression mapping learned from outcomes, PCA, regression and response classifiers must use training patients only.

### Expression prediction

Report for the world model and expression-output baselines:

- MSE and MAE on a common declared scale;
- correlation between predicted and observed post-treatment expression;
- correlation between predicted and observed expression change;
- predicted-versus-observed change magnitude;
- differential-expression direction agreement where appropriate.

### Clinical response

Report AUROC, AUPRC, balanced accuracy and MCC separately for:

- scGPT pretreatment embedding;
- frozen world-model pretreatment embedding;
- pretrained world-model source plus projected-transition embedding;
- paired-supervised fine-tuned world model;
- pretreatment PCA-expression classifier.

The paired-supervised world model receives post-treatment expression from training patients, whereas the scGPT embedding classifier does not. This difference must be disclosed.

## Pretreatment-only TCGA analysis

The TCGA workflow selects one primary, non-FFPE tumor profile per patient, aligns raw counts to checkpoint gene order and runs the fixed checkpoint without TCGA fine-tuning. It produces:

- pretreatment/source latent features;
- model-projected transition features;
- expression-change and latent-change magnitudes;
- empirical immune scores on observed and model-projected expression;
- exploratory OS and candidate-PFS Cox associations;
- within-cancer high/low stratification;
- exploratory associations with heterogeneous clinical-response records.

The model output must be described as a **model-projected transition**, not a confirmed post-treatment state or a causal treatment prediction. A primary TCGA tumor is not automatically proven to have been collected before every therapy; prior-treatment metadata and endpoint derivation flags require review.

## Raw-count and scale policy

The world model receives checkpoint-aligned raw counts with no CPM, log transformation, clipping or composition rescaling. Duplicate source genes mapping to one checkpoint gene are summed; missing model genes are zero-filled.

Many external perturbation tools require library normalization and log transformation. Their predictions must not be compared directly with raw-count predictions using MSE. Report either:

1. raw-count metrics only for methods operating on the raw-count scale; and
2. a separate common normalized/log-expression evaluation for all eligible methods.

Tool-required internal preprocessing must be declared rather than described as part of the world-model raw-count pipeline.

## Recommended external-model order

1. Keep identity, mean-delta and PCA-ridge.
2. Add reduced-rank regression.
3. Adapt scGen as the first external expression-output model.
4. Add CellOT only as an explicitly distribution-level sensitivity analysis.
5. Consider CPA only if the number of patients per regimen and treatment metadata support it.

## Why the current test run is slow

The visible `qnorm.quantile_normalize.py` messages originate from the COMPASS IFNG score. IFNG performs quantile normalization over the full patient-by-gene expression matrix. In the previous adapter, a float64 normalization result was assigned into a float32 pandas table, causing one warning for many gene columns. Terminal and log output from thousands of repeated warnings can dominate runtime.

The adapter now converts only the COMPASS scoring table to float64 before scoring. This does not change the raw-count values, but it prevents the dtype-warning flood on a newly started run. Quantile normalization itself remains computationally more expensive than simple marker averaging.

Other expected costs are:

- loading and aligning large raw-count matrices;
- patient-grouped world-model fine-tuning for every fold;
- scGPT embedding inference over up to `max_length` gene tokens;
- running every COMPASS method over multiple layers and cancer/treatment groups;
- TCGA inference and survival analysis across projects.

Test mode is intended to verify execution. Full-mode biological results should be interpreted only after gene-coverage, cohort-size, endpoint and treatment-history audits pass.
