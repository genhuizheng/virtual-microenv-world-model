# Pretreatment-only spatial biological prediction

This pipeline applies the fixed expression world model to pretreatment spatial
transcriptomics. It predicts a hypothetical post-treatment state and analyzes
the biological differences between pretreatment and predicted post-treatment.
Measured post-treatment H5AD files are not loaded or used.

## Scientific contract

- Stored H5AD `X` remains raw counts.
- Source genes are mapped into the checkpoint's exact gene order.
- The checkpoint-declared preprocessing is applied in memory. The corrected
  Gaussian-VAE checkpoint uses library-size normalization to 10,000 plus
  `log1p`.
- Decoder output is predicted `log1p_10k` expression, not integer raw counts.
- Predicted post-treatment expression and predicted changes remain attached to
  pretreatment coordinates.
- There is one prediction per pretreatment sample; measured post-treatment data
  are not loaded or used for any purpose in this workflow.
- The fixed checkpoint is not fine-tuned on spatial data.

## Dataset filter

The manifest uses the strict rule `n_genes > 10000`. Non-integer matrices are
excluded because stored raw counts are required. GeoMx AOIs, spatial spots/bins,
and single-cell spatial assays are summarized separately by platform.

## Biological analyses

- predicted gene-level changes and direction consistency across patients;
- genes whose predicted changes differ between response groups;
- COMPASS immune and ICI-response signature changes, including GEP, IFNG, CTL,
  CD8, TIDE, IMPRES, CAF, TAM and related scores;
- spatial maps of latent transition magnitude;
- Moran's I for spatially organized predicted gene changes;
- compartment-aware summaries using `segment_type` (or `region_type`) so that
  macrophage, vascular and tumor-region programs are not averaged together;
- Reactome preranked enrichment globally and by dataset, response group and
  segment, requiring at least 20% pathway gene coverage;
- a focused MATINS/GSE240138 mechanism panel covering STAB1/TFRC macrophage
  activation, IFN-gamma/TCR/CD28 signaling, antigen presentation, cytotoxicity
  and lipid-associated macrophage programs;
- an osimertinib-resistance panel for GSE288758 covering IFITM3, MET,
  PI3K-AKT, EGFR, TNF/IL6/IFNG signaling, proliferation and survival programs;
- spatial tests relating predicted IFITM3 changes in annotated tumor cells to
  predicted cytokine expression in neighboring observations and local
  MET/PI3K-AKT changes;
- a SARC037/GSE322640 Ewing-sarcoma panel using the paper's ten established
  EWS::FLI1-induced targets, with a baseline-expression-matched control score;
- dataset-, cancer-, platform-, patient- and response-stratified summaries.

## Outputs

- `spatial_baselines.tsv`: pretreatment-only prediction manifest.
- `sample_metrics.tsv`: transition magnitude, coverage and spatial organization.
- `gene_tables/*.tsv.gz`: pretreatment, predicted post-treatment and predicted
  delta means in checkpoint gene order.
- `latent/*.npz`: pretreatment coordinates and source/predicted latent states.
- `predicted_h5ad/*.h5ad`: top 50 predicted gene changes at every pretreatment
  coordinate. `X` is predicted post-treatment, `layers["pretreatment"]` is the
  input state and `layers["predicted_delta"]` is their difference.
- `biology_scores/`: COMPASS scores before prediction, after prediction, and
  predicted score changes.
- `segment_gene_tables/`: pretreatment, predicted post-treatment and delta
  expression summarized separately for each annotated spatial compartment.
- `reactome/`: model-predicted pathway enrichment with the exact GMT checksum
  recorded for reproducibility.
- `analysis/`: gene, signature, hotspot, response and cohort summaries.

## MATINS example and Reactome setup

GSE240138 is analyzed as a motivating example. The source study separated
CD68-positive macrophage, CD31-positive vascular and remaining tumor regions.
This pipeline uses existing standardized `segment_type`/`region_type`
annotations; it does not infer a cell type that is absent from the input.
The `test` batch mode selects all GSE240138, GSE288758 and GSE322640 baseline
patients but limits each sample to 256 observations. `full` mode analyzes every
eligible baseline dataset and every observation.

## Osimertinib-resistance example

The GSE288758 cohort contains four separately standardized baseline patients:
two response and two nonresponse patients. Its response comparisons are useful
as exploratory checks but are underpowered and must not be presented as
validated biomarkers. The pipeline reports predicted IFITM3-MET-PI3K/AKT
changes, tumor-compartment summaries, spatial cytokine associations and
within-dataset response associations.

GSE301973 is also associated with the paper, but the current local standardized
files have missing patient/timepoint metadata and slide-level files can contain
multiple patients. It is therefore intentionally excluded until it is
re-standardized into one correctly identified baseline sample per patient. This
prevents patient mixing and false patient-level inference.

## EWS::FLI1-reversal example

GSE322640 contributes five separately standardized Ewing-sarcoma baseline
patients. The paired measured post-treatment files are deliberately ignored.
The ten-gene induced-target panel is `NR0B1`, `IL1RAP`, `EZH2`, `CAV1`, `AKAP7`,
`CACNB2`, `RCOR1`, `FCGRT`, `FEZF1` and `FOCAD`. For every spatial observation,
the pipeline subtracts the mean of baseline-expression-matched control genes
from the target-gene mean. A negative predicted score change is labeled
predicted EWS::FLI1 program reversal. This is a model-derived hypothesis, not
evidence that EWS::FLI1 protein activity was experimentally suppressed.

Prepare the official Reactome GMT once on TACC:

```bash
mkdir -p /scratch/10119/ghzheng/Virtual_microenv/external/reactome
cd /scratch/10119/ghzheng/Virtual_microenv/external/reactome
wget -O ReactomePathways.gmt.zip https://reactome.org/download/current/ReactomePathways.gmt.zip
unzip -o ReactomePathways.gmt.zip
```

The paper used Reactome version 78. The default URL supplies the current
Reactome release, so the pipeline records the GMT SHA-256 checksum. Supply a
version-78 GMT through `REACTOME_GMT` when exact pathway-version matching is
required.

References: [MATINS spatial study](https://pmc.ncbi.nlm.nih.gov/articles/PMC10772343/),
[GSE240138](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE240138), and
[Reactome current downloads](https://reactome.org/download/current/). The second
example is the [IFITM3-MET osimertinib-resistance study](https://pmc.ncbi.nlm.nih.gov/articles/PMC12570849/)
with [GSE301973 metadata](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE301973).
The third example is the [SARC037 Ewing-sarcoma trial](https://pmc.ncbi.nlm.nih.gov/articles/PMC13321597/),
which deposited spatial data as GSE322640.

These results generate biological hypotheses from a generic transition model.
They are not measured treatment effects, treatment-specific causal predictions,
or evidence that a patient will respond.
