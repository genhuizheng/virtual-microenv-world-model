# Paired MOSCOT h5ad Dataloader Notes

## Purpose

This document records the assumptions and usage of the paired MOSCOT h5ad dataloader for test training.

The dataloader is designed for the sampled paired-expression dataset generated from MOSCOT transport pairs. Each training example is one source-to-target paired cell transition.

The intended learning task is:

```text
source / baseline gene expression  ->  target / posttreatment gene expression
```

where the MOSCOT transport probability can be used as a pair-level training weight.

---

## Files involved

### Sampling / paired h5ad construction script

```text
03_resample_pairs_build_paired_h5ad.py
```

This script samples MOSCOT pair rows and materializes source/target gene expression into paired h5ad shards.

### Dataloader script

```text
sample_paired_h5ad_dataloader.py
```

This script loads the paired h5ad shards and returns PyTorch-ready batches.

---

## Expected dataset folder

For the 50k test dataset, the expected folder is:

```text
paired_training_h5ad_50k/
  paired_moscot_pairs.tsv
  paired_pair_file_summary.tsv
  paired_h5ad_manifest.tsv
  paired_h5ad_global_summary.tsv
  genes.tsv
  shards/
    paired_expr_shard_00000.h5ad
    paired_expr_shard_00001.h5ad
    ...
```

The dataloader reads:

```text
paired_training_h5ad_50k/paired_h5ad_manifest.tsv
paired_training_h5ad_50k/shards/*.h5ad
```

The gene metadata is loaded from the first shard and should match `genes.tsv`.

---

## Paired h5ad format

Each shard is an AnnData object.

```text
adata.X
```

Stores the source / baseline expression matrix.

```text
adata.layers["target"]
```

Stores the target / posttreatment expression matrix.

```text
adata.obs
```

Stores paired-cell metadata.

```text
adata.var
```

Stores unified gene metadata.

---

## Row and column meanings

### Rows

Each row is one MOSCOT paired training example.

A row is not a single biological cell. It is a paired source-to-target transition:

```text
source_cell_original_obs -> target_cell_original_obs
```

### Columns

Each column is one unified gene.

The expected number of genes is:

```text
46,143 genes
```

The gene order is shared between source and target expression:

```text
adata.X[:, j] corresponds to the same gene as adata.layers["target"][:, j]
```

Therefore:

```text
batch["x"][:, j] and batch["y"][:, j] represent the same gene j
```

---

## Dataloader batch format

Each batch returned by the dataloader is a dictionary:

```python
batch["x"]            # source / baseline expression, shape [B, G]
batch["y"]            # target / posttreatment expression, shape [B, G]
batch["weight"]       # pair-level training weight, shape [B]
batch["probability"]  # raw MOSCOT probability, shape [B]
```

Additional metadata fields are also returned:

```python
batch["pair_id"]
batch["patient_id"]
batch["dataset_id"]
batch["transition"]
batch["source_cell"]
batch["target_cell"]
```

For the 50k test dataset with batch size 64, the expected shapes are:

```text
batch["x"].shape      = [64, 46143]
batch["y"].shape      = [64, 46143]
batch["weight"].shape = [64]
```

---

## Gene metadata

Gene information is important and should be preserved outside the tensor.

The dataloader stores gene-level metadata as:

```python
dataset.gene_ids
dataset.gene_symbols
dataset.gene_var
dataset.get_gene_info()
```

The correct interpretation is:

```text
batch["x"][:, j] and batch["y"][:, j]
    correspond to dataset.gene_ids[j] / dataset.gene_symbols[j]
```

Example first genes:

```text
ENSG00000000003 / TSPAN6
ENSG00000000005 / TNMD
ENSG00000000419 / DPM1
ENSG00000000457 / SCYL3
ENSG00000000460 / C1orf112
```

### Why gene names are not stored inside every batch by default

The expression tensor should contain only numeric gene expression values.

Gene names are fixed for all batches and are therefore stored once at the dataset level. Repeating 46,143 gene names inside every batch is unnecessary and inefficient.

For debugging only, gene names can be included in each batch using:

```python
include_gene_info_in_batch=True
```

---

## Probability usage assumption

The MOSCOT probability represents the strength of the source-to-target transport link.

In the dataloader, probability can be used in two possible ways.

### Recommended first test setting

```python
sample_by_probability=False
use_probability_as_weight=True
```

Meaning:

```text
Rows are sampled/shuffled uniformly from the paired h5ad shards.
The MOSCOT probability is used as a loss weight.
```

This is recommended for the first training test because the 50k paired dataset was already sampled from MOSCOT pairs, and using probability again for sampling may over-emphasize high-probability pairs.

### Alternative later setting

```python
sample_by_probability=True
use_probability_as_weight=True
```

Meaning:

```text
High-probability pairs appear more often during training.
The probability is also used as the loss weight.
```

This may be useful later, but it can double-emphasize high-probability pairs.

---

## Weighted loss assumption

For first test training, use pair-level weighted MSE:

```python
per_pair_loss = ((pred - target) ** 2).mean(dim=1)
loss = (per_pair_loss * weight).sum() / (weight.sum() + 1e-8)
```

Here:

```text
pred   = model(batch["x"])
target = batch["y"]
weight = batch["weight"]
```

The weight applies to the full gene-expression vector for each paired example.

---

## Example dataloader usage

```python
from sample_paired_h5ad_dataloader import build_paired_h5ad_loader, weighted_mse_loss

dataset, loader = build_paired_h5ad_loader(
    data_dir="paired_training_h5ad_50k",
    batch_size=64,
    sample_by_probability=False,
    use_probability_as_weight=True,
    num_workers=0,
)

print(dataset.n_genes)
print(dataset.gene_ids[:10])
print(dataset.gene_symbols[:10])

for batch in loader:
    x = batch["x"]
    y = batch["y"]
    w = batch["weight"]

    pred = model(x)
    loss = weighted_mse_loss(pred, y, w)

    loss.backward()
    break
```

---

## Smoke-test command

```bash
python sample_paired_h5ad_dataloader.py \
  --data-dir paired_training_h5ad_50k \
  --batch-size 64
```

Expected output:

```text
n_genes: 46143
x shape: (64, 46143)
y shape: (64, 46143)
weight shape: (64,)
```

---

## Training smoke-test command

```bash
python sample_paired_h5ad_dataloader.py \
  --data-dir paired_training_h5ad_50k \
  --batch-size 64 \
  --train-smoke-test
```

Expected behavior:

```text
The script should load batches, run model forward, compute weighted loss,
run backward, and update the optimizer.
```

The dummy smoke-test model is not biologically meaningful. It only verifies that the pipeline is technically functional.

---

## Confirmed smoke-test result

The dataloader smoke test confirmed:

```text
n_genes = 46143
first 10 gene_ids are available
first 10 gene_symbols are available
approx batches per epoch = 782
sample_by_probability = False
```

The first batch confirmed:

```text
x shape = (64, 46143)
y shape = (64, 46143)
weight shape = (64,)
probability is available
```

The training smoke test also completed forward/backward/optimization steps.

---

## Current assumptions

1. All paired h5ad shards use the same unified gene order.
2. `adata.X` is source / baseline expression.
3. `adata.layers["target"]` is target / posttreatment expression.
4. `adata.obs["probability"]` is the MOSCOT pair probability.
5. Probability is first used as a pair-level loss weight, not as dataloader sampling weight.
6. Gene metadata is fixed across batches and stored at the dataset level.
7. The 50k dataset is for test training only.
8. Larger-scale training can later use the same format with 500k or more pairs.

---

## Recommended next step

Use the 50k paired dataset to test a real transition model.

Start with:

```text
sample_by_probability = False
use_probability_as_weight = True
batch_size = 64 or 128
```

If the model trains stably, scale to:

```text
paired_training_h5ad_500k
```

with larger shard size and possibly multi-worker loading after confirming I/O stability.
