# Future Record 02: 500k Training And Validation

## Goal

Scale from the 50k paired h5ad screening dataset to a 500k paired dataset with a clear train/validation split.

The purpose is to check whether the selected model continues to improve when more MOSCOT-paired transitions are available.

## Dataset Plan

Expected folder:

```text
paired_training_h5ad_500k/
  paired_h5ad_manifest.tsv
  genes.tsv
  shards/
    paired_expr_shard_00000.h5ad
    paired_expr_shard_00001.h5ad
    ...
```

Recommended split:

```text
50k model selection: 5-fold shard-level cross-validation
500k later scale-up: shard-level train/validation split or 5-fold if compute allows
```

Shard-level splitting is preferred because it avoids mixing rows from the same shard across train and validation.

## Training Defaults

Start from the best 50k model level.

Suggested GH200/H200 settings:

```text
batch_size: 512 or 1024
epochs: 5 to 20
model_dim: 256
depth: 6
heads: 4
num_tokens: 4
loss_mode: chreode
```

For the current MOSCOT-paired screening design, use:

```text
loss_mode: pair_mse
primary selection metric: val_loss_pair
secondary metrics: val_metric_mmd, val_metric_w2, val_metric_drift
```

If memory is comfortable, test:

```text
batch_size: 1024
sinkhorn_iters: 100
```

If training is slow, start with:

```text
batch_size: 512
sinkhorn_iters: 50
```

## Validation Outputs

Each run should save:

```text
checkpoints_level{level}/train_epoch_losses.csv
checkpoints_level{level}/val_epoch_losses.csv
checkpoints_level{level}/world_model_level{level}_step{step}.pt
checkpoints_level{level}/best_model_level{level}_fold{fold}.pt
checkpoints_level{level}/run_config.json
```

Validation should report:

```text
loss
loss_pair
metric_mmd
metric_w2
metric_drift
metric_down
```

## Success Criteria

The model is promising if:

```text
training loss decreases across epochs
validation loss decreases or remains stable
no NaN or divergence occurs
the selected level has the lowest mean val_loss_pair across folds
```

## Current Code Support

The current training script supports:

```text
shard-level train/validation fold split
--num-folds
--fold-index
separate train_epoch_losses.csv and val_epoch_losses.csv
best-checkpoint saving by val_loss_pair
```
