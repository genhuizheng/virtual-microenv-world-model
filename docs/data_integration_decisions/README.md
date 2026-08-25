# Data Integration / Batch-Effect Decision Log

## Context

Main data resource: [scCT-DB](https://academic.oup.com/nar/article/54/D1/D1616/8306130) — 266 paired
pre/post-treatment scRNA-seq datasets, 6.19M cells, 27 cancer types, 5
sequencing technologies (10x, MARS-seq, BD Rhapsody, Microwell-seq, Seq-Well),
uniformly processed but pooled from many labs/studies.

Problem: the Stage-1 scVI reconstruction loss (~5000-6000 on 46,143 genes) has
two causes worth separating:

1. Scale artifact — NB log-likelihood summed over 46k genes vs. the ~2k HVGs
   typical scVI tutorials use. Not inherently wrong.
2. Genuine batch effect — real cross-study/cross-technology technical
   variation, plus dead-weight zero-filled genes from the union-based unified
   gene panel (`batch_processed_unified_genes_zero_fill`).

## Metadata recovery (resolved 2026-08-25)

`technology`/`platform` were missing from the training data because
`Code/preprocessing/batch_processing.py` writes them to
`adata.uns["metadata"]` (lines 379-380), not to `adata.obs` -- and the pairing
step builds shards from `obs`. The original `Sample_info/*/Sample info.txt`
tables are **gone** (all 266 directories exist but are empty), but the values
survived in `uns["metadata"]` of all 568 per-patient/dataset h5ads in both
`batch_processed/` and `batch_processed_unified_genes_zero_fill/`.

Scanned result (50k pair set):

| technology | n_pairs | pct |
|---|---:|---:|
| 10X Genomics | 44,722 | 89.44 |
| 10x Genomics | 3,869 | 7.74 |
| BD Rhapsody | 909 | 1.82 |
| Microwell-seq | 225 | 0.45 |
| Seq-Well | 170 | 0.34 |
| MARS-seq | 105 | 0.21 |

Three consequences:

1. **`10X Genomics` and `10x Genomics` are the same platform**, differing only
   in capitalization (460 vs 47 directories). Combined = 97.18% of pairs. Any
   use of `technology` as a batch key MUST normalize case first, or scVI will
   treat one platform as two batches and "correct" a non-existent difference.
2. **Dropping non-10x costs 2.82% of pairs** (1,409 / 50,000). Option #1 is
   cheap in volume; the open question is whether those 4 studies carry unique
   cancer types/regimens.
3. **`source_study` is the right technical batch unit and is already
   available.** The patient directory embeds the source accession
   (`E-MTAB-12733-Pt1`, `GSE246613-...`), giving **65 distinct source studies**
   -- exactly the "65 eligible source datasets" in the scCT-DB paper. Because
   `paired_moscot_pairs.tsv` already carries that same string in its
   `patient_id` column, `source_study` can be derived at load time with a
   regex: no shard rebuild, no join, no reprocessing.

Candidate batch groupings compared:

| Grouping | Categories | Assessment |
|---|---:|---|
| `dataset_id` | 266 | Partitioned by therapeutic regimen, cancer subtype, and drug-response group -- **biologically confounded**, would erase modeled signal |
| `technology` | 5 (→1 after #1) | Too coarse, and degenerate once non-10x is dropped |
| `source_study` | 65 | One study = one lab = one platform. Purely technical. **Preferred** |

Also verified: 0 scCT datasets span more than one technology (metadata is
internally consistent), 266 datasets / 408 patients / 65 source studies.

## Active options (in priority order)

| # | Option | Status |
|---|---|---|
| 1 | Restrict to dominant platform only (drop minority non-10x studies) | **Unblocked** — costs 2.82% of pairs (see metadata recovery above). Implement as a load-time row filter keyed on the recovered technology lookup; no shard rebuild needed |
| 2 | scVI `batch_key` on **`source_study`** (revised from `dataset_id`) | Next — 65 clean technical categories, derivable at load time by regex on the existing `patient_id` column. `scvi_stage1_representation.py` already accepts per-cell batch categories; `train_stage1_scvi.py` CLI + threading through `five_level_cell_world_model.py`/`train_five_level_world_model.py` still pending. **Do not use `dataset_id`** — it is partitioned on biological variables |
| 3 | Gene intersection instead of union+zero-fill | Reconsider after seeing results of 1+2 — changes `n_genes`, so worth bundling into the same Stage-1 rebuild rather than redoing it twice |
| 4 | scGPT-style value-binned input, continuous output | Kept active as a parallel/fallback representation, **not** the full scGPT foundation model. Per-cell rank/quantile binning of expression (new `expression_transform` option in `sample_paired_h5ad_dataloader.py`) feeding the existing `GaussianExpressionVAE` (continuous MSE decoder, not scVI's NB likelihood, since binned values aren't counts). Batch-robust input, DEG/logFC-compatible output. Try this if 1-3 don't move the reconstruction loss enough, or as an independent comparison against the scVI-batch-key approach. |
| 5 | trVAE-style MMD batch-overlap penalty | **Cheapest new addition** — pairs with #2 rather than replacing it. A batch-conditional VAE only *implicitly* absorbs batch; trVAE adds an explicit MMD penalty in the loss that directly forces latent overlap between batches. `chreode_loss.py` already implements multi-bandwidth RBF MMD (`rbf_mmd`), currently used for predicted-vs-target population matching — the same function can penalize MMD between `dataset_id` groups during Stage-1 training. Near-zero new code, no new dependency. |
| 6 | scVI `batch_key` on **`platform`/`technology`** | Alternative granularity for #2, same code path — only the column name changes, so implementing #2 gets this for free. Corrects the dominant (cross-platform) source of technical variance using a small closed set (~5 categories: 10x, MARS-seq, BD Rhapsody, Microwell-seq, Seq-Well) instead of an unbounded one. **Future-proof**: a genuinely new study on an existing platform is a *seen* category, whereas a new `dataset_id` is always unseen. Trade-off: coarser than `dataset_id`, so it leaves per-study/per-lab quirks within a platform uncorrected. |

### Interaction between #1 and #6 — resolved by using `source_study`

#1 and #6 conflict directly: dropping non-10x makes `technology` a constant, so
a `technology` batch key becomes the single-dummy-batch case and corrects
nothing. `#1 + #6` is self-defeating.

The recovered metadata dissolves the conflict, because `source_study` (65
categories) survives the platform filter — roughly 61 studies remain after
dropping the 4 non-10x ones, all still technical and still multi-category.

**Recommended combination: #1 + #2-on-`source_study`.**

| Combination | Meaning |
|---|---|
| **#1 + #2 (`source_study`)** | **Preferred.** Drop non-10x (2.82% of pairs), then correct residual per-study/per-lab effects with a 65→~61-category technical batch key. No biological confounding, no degenerate batch key. |
| #6 alone (`technology`) | Keep all platforms, correct across the 5 (case-normalized) technologies. Retains the 2.82%, but coarse — leaves within-platform lab effects uncorrected. |
| #1 alone | Drop non-10x, no batch key at all. Simplest; leaves within-10x lab effects uncorrected. Useful as an ablation against #1+#2. |

Note `source_study` shares `dataset_id`'s unseen-category limitation for
genuinely new studies at inference — that is what parked scPoli-style learnable
batch embeddings would eventually address.

## Parked (not pursuing now, revisit only if 1-6 aren't enough)

- scPoli-style **learnable batch embeddings** — instead of a fixed one-hot batch category, each batch gets a small learned embedding vector; the review highlights this as built specifically for "efficient transfer to new datasets". A direct upgrade to #2 that partially answers the *novel-`dataset_id`-at-inference* problem (a new batch can fit a fresh embedding without full retraining) — lighter than scArches but a real implementation step, so parked until #1-#2 results are in.
- Read-depth-aware conditioning (scFoundation, scPRINT) — condition on sequencing depth explicitly; largely redundant with scVI's internal library-size term, so low priority.
- Other batch approaches catalogued by the 2026 review but not pursued: CLAIRE (contrastive learning with mutual-nearest-neighbour batch overlap), LEMUR (batch-dependent linear subspace), SATURN (cross-species macrogenes via protein language models), Scanorama (post-hoc linear correction, same class as Harmony).
- scArches incremental/query mapping — only matters once genuinely new datasets arrive after Stage 1 is trained
- Harmony/ComBat upstream correction — Harmony is plausible at 6M cells (operates in PCA space, ~1.2GB for 6M x 50 PCs; scCT-DB's own paper used it) but runtime-heavy on the standard CPU-bound `harmonypy` implementation; ComBat is riskier at this scale (fits per-gene across all cells, designed for microarray-era sample counts). Reprojecting a Harmony-corrected embedding back to gene expression via inverse-PCA is a lossy, ungrounded hack (low-rank reconstruction, no learned likelihood, no guarantee of valid output) — not a substitute for a trained decoder.
- Full scGPT foundation model — paused by explicit decision: focus on the self-built VAE first; revisit only if it works extremely well and a foundation-model upgrade still seems worth the gene-vocabulary-alignment and decoder-mismatch cost
- Adversarial/CPA-style batch-invariant latent (gradient reversal) — upgrade path if scVI's conditional-decoder approach underperforms; actively strips batch signal from the latent instead of conditioning the decoder on it, which also sidesteps the "predicted cell has no natural batch label" awkwardness
- Reference-atlas mapping to an existing published batch-corrected model — never seriously scoped (would need a compatible atlas: species, cancer types, gene panel)

## What the reference paper (Chreode) actually does

Checked directly against its methods section (arXiv 2605.28111), not assumed:

- scVI `batch_key` = `leaf_dataset` (their `dataset_id` equivalent) — validates option #2 directly.
- Deliberately excludes the biological variable of interest (their developmental
  timepoint; our treatment status) from the batch covariate — never let batch
  correction erase the signal being modeled.
- Expression transform: `normalize_total(1e4)` + `log1p` — already matches this
  project's existing `log1p_10k` transform.
- Restricts to genes shared across all source datasets (16,520 mouse-human
  orthologs, filtered from ~33k) rather than a zero-filled union — validates
  option #3 directly.

Sources: [arXiv abstract](https://arxiv.org/abs/2605.28111), [full text](https://arxiv.org/html/2605.28111v1)

## Related literature: large-scale scRNA-seq representation / compression

- **scVI** (Lopez et al. 2018, *Nature Methods*) — the foundational
  count-based VAE for scRNA-seq; what Stage 1 in this project is built on.
- **scGPT** (Cui et al. 2024, *Nature Methods*) — transformer foundation
  model using rank/bin-based expression tokenization plus pretrained gene
  embeddings; the basis of the parked "full scGPT" option and the active
  "value-binned input" option (#4).
- **Geneformer** (Theodoris et al. 2023, *Nature*) — single-cell foundation
  model pretrained on ~30M cells, also using rank-based gene-value encoding
  per cell (conceptually close to scGPT's binning, independently motivated).
- **UCE / Universal Cell Embeddings** (Rosen et al. 2023-2024) — cross-species,
  cross-study cell embeddings built from protein-sequence-based gene
  representations rather than a shared gene vocabulary.
- **DCA — deep count autoencoder** ("Massive single-cell RNA-seq analysis and
  imputation via deep learning" preprint; published as Eraslan et al. 2019,
  *Nature Communications*) — an autoencoder explicitly aimed at denoising and
  compressing large-scale scRNA-seq count data.
- **"Representation learning of single-cell RNA-seq data"** (2026 review,
  [RNA journal](https://rnajournal.cshlp.org/content/32/4/504.full) /
  [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC12990802/)) — current survey
  of autoencoder/VAE/GNN/transformer approaches to compressing scRNA-seq data,
  motivated explicitly by public repositories now holding 100M+ cells.

Sources: [Representation learning of single-cell RNA-seq data (RNA journal)](https://rnajournal.cshlp.org/content/32/4/504.full), [same, PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC12990802/), [Massive single-cell RNA-seq analysis and imputation via deep learning (bioRxiv preprint)](https://www.biorxiv.org/content/10.1101/315556.full.pdf)

**Important negative finding from that review:** it does *not* benchmark which
methods work best at millions-of-cells scale. It notes 100M+ cells exist in
public repositories and that foundation models train on tens of millions, but
reports no direct performance-versus-scale comparison, explicitly calling
systematic cross-family benchmarking an open gap. So none of the options above
can be selected from the literature alone — they have to be tested on this
project's own data.
