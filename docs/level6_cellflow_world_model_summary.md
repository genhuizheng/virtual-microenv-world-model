# Level 6 CellFlow World Model Summary

## Purpose

This project trains a single-cell treatment-response world model.

The model learns the transition:

```text
pretreatment / baseline cell state -> post-treatment cell state
```

The current final model is the **Level 6 CellFlow-style flow-matching model**.
Only this flow-matching architecture is summarized here.

## Fixed Gene Space

All training datasets must use the same fixed gene space.

Requirement:

```text
same genes
same gene order
same input/output dimension
```

The current data pipeline uses a unified zero-filled gene matrix:

```text
batch_processed_unified_genes_zero_fill
```

If a gene is missing from one dataset, that gene is filled with zero for that
dataset. This lets multiple cohorts share one common expression space.

The model input and output dimensions are therefore:

```text
n_genes = number of unified fixed genes
```

## Training Data Format

The training data are paired AnnData `.h5ad` shards generated from MOSCOT
source-target pairings.

Example directory:

```text
paired_training_h5ad_1000k_fast/
  paired_h5ad_manifest.tsv
  paired_h5ad_global_summary.tsv
  genes.tsv
  shards/
    paired_expr_shard_00000.h5ad
    paired_expr_shard_00001.h5ad
    ...
```

Each shard is one AnnData object.

### AnnData Fields

```text
adata.X
```

Source expression matrix.

```text
shape = [n_pairs_in_shard, n_genes]
meaning = pretreatment / baseline cells
```

```text
adata.layers["target"]
```

Target expression matrix.

```text
shape = [n_pairs_in_shard, n_genes]
meaning = post-treatment cells
```

```text
adata.obs
```

Pair-level metadata.

Expected metadata include:

```text
probability        MOSCOT transport probability / pair weight
patient_id         patient identifier, if available
dataset_id         source dataset identifier
transition         treatment transition label
source_cell        source cell identifier
target_cell        target cell identifier
```

```text
adata.var
```

Fixed gene metadata.

Expected gene metadata include:

```text
gene_id
gene_symbol
```

## Dataloader Batch Format

The PyTorch dataloader yields one batch dictionary:

```text
batch["x"]
```

Source / pretreatment expression.

```text
shape = [batch_size, n_genes]
```

```text
batch["y"]
```

Target / post-treatment expression.

```text
shape = [batch_size, n_genes]
```

```text
batch["weight"]
```

Pair weight from MOSCOT transport probability.

```text
shape = [batch_size]
```

```text
batch["delta"]
```

Elapsed treatment-time value. If no time column is provided, the default value is
used.

```text
shape = [batch_size]
```

## Model Architecture: Level 6 CellFlow-Style Flow Matching

The Level 6 model is a latent velocity-field model inspired by CellFlow.

The model has three main components:

```text
Expression encoder
Flow-matching velocity field
Expression decoder
```

### Encoder

The encoder maps fixed-gene expression into latent space:

```text
z_source = encoder(x_pre)
z_target = encoder(x_post)
```

Current latent dimension:

```text
latent_dim = 128
```

### Flow-Matching Velocity Field

The model learns a velocity field:

```text
v_theta(z_t, t, delta)
```

where:

```text
z_t     interpolated latent state between source and target
t       flow time, sampled from [0, 1]
delta   treatment time-gap condition, or default time step
```

For each paired training example:

```text
z_source = encoder(x_pre)
z_target = encoder(x_post)
t ~ Uniform(0, 1)
z_t = (1 - t) * z_source + t * z_target + sigma * noise
target_velocity = z_target - z_source
```

The velocity field is trained to predict:

```text
v_theta(z_t, t, delta) ~= z_target - z_source
```

No perturbation/action token is used in the current version.

### Decoder

The decoder maps predicted post-treatment latent state back into expression
space:

```text
y_pred = decoder(z_pred)
```

## Inference

At inference, the model starts from a pretreatment cell:

```text
x_pre
```

It encodes the cell:

```text
z_source = encoder(x_pre)
```

Then it integrates the learned velocity field from `t = 0` to `t = 1`:

```text
z_pred = integrate v_theta(z_t, t, delta)
```

Finally, it decodes the predicted post-treatment state:

```text
y_pred = decoder(z_pred)
```

## Model Input and Output

### Input

```text
x_pre
```

Pretreatment / baseline expression vector in the fixed gene space.

```text
shape = [n_genes]
```

For batch inference:

```text
shape = [batch_size, n_genes]
```

### Output

```text
y_pred
```

Predicted post-treatment expression vector in the same fixed gene space.

```text
shape = [n_genes]
```

For batch inference:

```text
shape = [batch_size, n_genes]
```

The model also outputs latent-level quantities:

```text
z_source   source latent state
z_pred     predicted post-treatment latent state
velocity   learned treatment transition velocity
residual   z_pred - z_source
```

## Training Objective

The primary Level 6 training loss is flow matching:

```text
loss_cellflow =
weighted MSE(
  v_theta(z_t, t, delta),
  z_target - z_source
)
```

The pair weights come from MOSCOT transport probabilities:

```text
weight = batch["weight"]
```

The final training run also supports auxiliary endpoint/population terms:

```text
pair auxiliary loss:       z_pred vs z_target
population auxiliary loss: MMD/W2/downstream population terms
```

## Final Full-Data Training Setting

The final full-data training uses:

```text
data_dir = paired_training_h5ad_1000k_fast
level = 6
loss_mode = cellflow
batch_size = 2048
epochs = 50
lr = 1e-5
num_folds = 1
fold_index = 0
```

CellFlow-specific settings:

```text
cellflow_sigma = 0.01
cellflow_integration_steps = 8
cellflow_integration_method = euler
cellflow_pair_aux_weight = 0.1
cellflow_population_aux_weight = 1.0
```

Population auxiliary settings:

```text
lambda_mmd = 3e-4
lambda_w2 = 3e-3
lambda_drift = 3.0
lambda_down = 0.0
```

Output checkpoint directory:

```text
checkpoints_1000k_full_cellflow_level6_full_data
```

Main checkpoint:

```text
best_model_level6_fold0.pt
```

Here `fold0` only reflects `num_folds = 1`; it means full-data training, not
cross-validation fold 0.

## Essential Interpretation

The model learns a latent treatment-response velocity field from MOSCOT-paired
pre/post-treatment cells.

In simple terms:

```text
input:  pretreatment cell expression
output: predicted post-treatment cell expression
```

Because input and output both use the same fixed gene space, predicted treatment
changes can be analyzed at gene, pathway, cell-state, patient, or cohort level.
