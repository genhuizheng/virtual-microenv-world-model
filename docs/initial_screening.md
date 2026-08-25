# Initial Screening Summary

## Goal

Select the best first-round world-model level on the 50k MOSCOT-paired h5ad dataset.

This is a screening experiment, not the final scVI-based Chreode reproduction.

## Data

```text
dataset: paired_training_h5ad_50k
examples: MOSCOT paired source-target rows
x: source / baseline expression
y: target / posttreatment expression
weight: MOSCOT transport probability
delta: 1.0 by default
```

## Validation Design

```text
split: 5-fold shard-level cross-validation
train: 4 folds
validation: 1 fold
selection metric: validation loss_pair
```

Each level is trained on all five folds:

```text
fold_index = 0, 1, 2, 3, 4
```

## Current Training Loss

Primary training loss:

```text
loss = loss_pair
```

where:

```text
loss_pair = weighted MSE(z_pred, z_target)
```

The weight is:

```text
batch["weight"] = MOSCOT probability
```

This means the model is trained to predict the MOSCOT-paired target latent state.

Level 6 exception:

```text
--level 6 --loss-mode cellflow
```

For Level 6, the primary training loss is CellFlow-inspired flow matching:

```text
t ~ Uniform(0, 1)
z_t = (1 - t) z_source + t z_target
loss_cellflow = weighted MSE(v_theta(z_t, t, delta), z_target - z_source)
```

The script still logs `loss_pair`, MMD, W2, and drift for comparison.

## Logged Metrics

The training script also logs secondary metrics:

```text
metric_mmd
metric_w2
metric_drift
metric_down
```

Definitions:

```text
metric_mmd:
  MMD(z_pred batch, z_target batch)

metric_w2:
  Sinkhorn W2(z_pred batch, z_target batch)

metric_drift:
  weighted MSE((z_pred - z_source), (z_target - z_source))

metric_down:
  downhill regularizer, active only for Level 4-style potential-flow models
```

For the current MOSCOT-paired setup:

```text
metric_drift is effectively equivalent to loss_pair
```

because:

```text
(z_pred - z_source) - (z_target - z_source) = z_pred - z_target
```

## Hyperparameters Used

```text
loss_mode: pair_mse
learning rate: 1e-5
batch size: 2048
epochs: 50
max_steps: -1
num_folds: 5
latent_dim: 128
model_dim: 256
depth: 4
heads: 8
num_tokens: 8
optimizer: AdamW
weight_decay: 0.01
grad_clip: 1.0
```

## Historical First-Pass Results

These were early screening results before the expanded 50-epoch rerun. Keep them only as historical context.

Ranking by final 5-fold validation `loss_pair`:

| Rank | Level | Model | Mean final val loss_pair | Verdict |
|---:|---:|---|---:|---|
| 1 | 4 | Waddington / Chreode-style residual DiT | ~0.000812 | Best |
| 2 | 3 | Gated one-step residual DiT | ~0.001738 | Good |
| 3 | 1 | Transformer residual model | ~0.001787 | OK |
| 4 | 2 | DiT residual model | ~0.002009 | Worse than expected |
| 5 | 5 | Advanced stochastic DiT prototype | ~0.003379 | Worst |

Do not mix this table with the new 50-epoch results.

## Added Level 5 Variant Screening

The code now supports explicit Level 5 variants through:

```text
--level 5 --variant <name>
```

Implemented for the current 50-epoch screening round:

| Label | Variant name | Current status |
|---|---|---|
| 5B | `tokenized_dit` | Implemented, deterministic tokenized AdaLN-Zero DiT |
| 5C | `stochastic_dit` | Implemented, stochastic residual DiT |
| 5D | `cross_attention_dit` | Implemented prototype with learned null context token |
| 5E | `hierarchical_dit` | Implemented hierarchical/U-Net-inspired token DiT |
| 5F | `consistency_dit` | Implemented one-step consistency-style prototype |
| 5G | `rectified_dit` | Implemented rectified-flow-inspired one-step displacement DiT |
| 5H | `diffusion_denoising_dit` | Not implemented for current screening |

The training loss is still unchanged:

```text
loss = weighted MSE(z_pred, z_target)
```

So these variants are directly comparable by `val_loss_pair`, as long as we compare runs with the same epoch count.

## Current 50-Epoch Folder Rules

Use only these folders for the 50-epoch architecture summary:

```text
default Levels 1-5:
  checkpoints_pair_lr1e5_epoch50_level*_fold*

Level 5 variants:
  checkpoints_pair_lr1e5_level5_tokenized_dit_fold*
  checkpoints_pair_lr1e5_level5_stochastic_dit_fold*
  checkpoints_pair_lr1e5_level5_cross_attention_dit_fold*
  checkpoints_pair_lr1e5_level5_hierarchical_dit_fold*
  checkpoints_pair_lr1e5_level5_consistency_dit_fold*
  checkpoints_pair_lr1e5_level5_rectified_dit_fold*
```

Exclude old default folders without `epoch50` in the name:

```text
checkpoints_pair_lr1e5_level1_fold*
checkpoints_pair_lr1e5_level2_fold*
checkpoints_pair_lr1e5_level3_fold*
checkpoints_pair_lr1e5_level4_fold*
checkpoints_pair_lr1e5_level5_fold*
```

Those may be 25-epoch runs and should not be mixed into the 50-epoch comparison.

## Current Selection Rule

For model architecture selection, average across the five folds.

Primary ranking metric:

```text
mean_best_val_loss_pair across folds
```

Stability check:

```text
mean_final_val_loss_pair
```

Interpretation:

```text
best epoch = lowest validation loss_pair within one run
final epoch = epoch 49 for a 50-epoch run
```

A good model should have low `mean_best_val_loss_pair` and a `mean_final_val_loss_pair` that is not much worse.

Important duplicate rule:

```text
level5 and level5_stochastic_dit are the same architecture in the current code.
```

For architecture selection, merge them as:

```text
level5_stochastic_dit -> level5
```

## Result Section: Top 5 Unique Architectures

The strict 50-epoch summary produced the following top 5 unique architectures.

| Rank by pair loss | Code label | Architecture name | Folds | Mean best pair | Mean final pair | Mean final MMD | Mean final W2 | Notes |
|---:|---|---|---:|---:|---:|---:|---:|---|
| 1 | `level1` | Plain Transformer Residual Model | 5 | 0.0006762947 | 0.0006762947 | 0.003754749 | 0.12701087 | Best pair loss and stable |
| 2 | `level4` | Waddington / Chreode-style Residual DiT | 5 | 0.00074958546 | 0.0015930883 | 0.027070568 | 0.22737993 | Good best loss, less stable final/population metrics |
| 3 | `level5` | Stochastic AdaLN-Zero DiT | 5 | 0.00082005708 | 0.00082005708 | 0.001956857 | 0.14712527 | Stable default stochastic Level 5 |
| 4 | `level5_hierarchical_dit` | Hierarchical / U-Net-inspired Token DiT | 5 | 0.00089149289 | 0.00089149289 | 0.0065097287 | 0.14696709 | Stable best non-default Level 5 variant |
| 5 | `level5_cross_attention_dit` | Cross-Attention DiT with Null Context Token | 5 | 0.00092059914 | 0.00092059914 | 0.0097930487 | 0.15731152 | Stable second-best non-default Level 5 variant |

Important:

```text
level5_stochastic_dit was removed from the top list because it duplicates level5.
```

`level5_consistency_dit` is excluded because it exploded:

```text
mean_best_pair = 0.085658008
mean_final_pair = 14041.897
mean_final_w2 = 2985.4103
```

These five architectures are selected for 500k retesting, with stability-aware priority:

```text
level1
level5
level5_hierarchical_dit
level5_cross_attention_dit
level4
```

`level4` remains in the 500k retest because its best loss is strong, but it should be watched carefully because its final loss, MMD, and W2 are worse than the other selected models.

500k retest dataset:

```text
paired_training_h5ad_500k_fast
```

500k retest command:

```bash
source /home1/10119/ghzheng/.bashrc
conda activate worldmodeltraining

cd /scratch/10119/ghzheng/Virtual_microenv

# 500k retest: top default architectures
for level in 1 4 5; do
  for fold in 0 1 2 3 4; do
    echo "===== 500k Retest Level $level Fold $fold Epoch 50 ====="
    python train_five_level_world_model.py \
      --data-dir paired_training_h5ad_500k_fast \
      --level "$level" \
      --batch-size 2048 \
      --epochs 50 \
      --max-steps -1 \
      --num-folds 5 \
      --fold-index "$fold" \
      --lr 1e-5 \
      --save-every 0 \
      --no-save-each-epoch \
      --sinkhorn-iters 20 \
      --checkpoint-dir "checkpoints_500k_retest_pair_lr1e5_epoch50_level${level}_fold${fold}"
  done
done

# 500k retest: top Level 5 variants
for variant in hierarchical_dit cross_attention_dit; do
  for fold in 0 1 2 3 4; do
    echo "===== 500k Retest Level 5 Variant $variant Fold $fold Epoch 50 ====="
    python train_five_level_world_model.py \
      --data-dir paired_training_h5ad_500k_fast \
      --level 5 \
      --variant "$variant" \
      --batch-size 2048 \
      --epochs 50 \
      --max-steps -1 \
      --num-folds 5 \
      --fold-index "$fold" \
      --lr 1e-5 \
      --save-every 0 \
      --no-save-each-epoch \
      --sinkhorn-iters 20 \
      --checkpoint-dir "checkpoints_500k_retest_pair_lr1e5_epoch50_level5_${variant}_fold${fold}"
  done
done
```

## Level 6 CellFlow-Inspired Screening Command

Run Level 6 separately because its primary loss is `cellflow`, not `pair_mse`.

```bash
source /home1/10119/ghzheng/.bashrc
conda activate worldmodeltraining

cd /scratch/10119/ghzheng/Virtual_microenv

for fold in 0 1 2 3 4; do
  echo "===== 50k Training Level 6 CellFlow Fold $fold Epoch 50 ====="
  python train_five_level_world_model.py \
    --data-dir paired_training_h5ad_50k \
    --level 6 \
    --loss-mode cellflow \
    --batch-size 2048 \
    --epochs 50 \
    --max-steps -1 \
    --num-folds 5 \
    --fold-index "$fold" \
    --lr 1e-5 \
    --checkpoint-dir "checkpoints_pair_lr1e5_epoch50_level6_cellflow_fold${fold}"
done
```

Selection metric for Level 6 checkpoints:

```text
val_loss_cellflow
```

Comparison metrics against Levels 1-5:

```text
loss_pair
metric_mmd
metric_w2
metric_drift
```

## TACC 50-Epoch Summary Command

Run from:

```bash
cd /scratch/10119/ghzheng/Virtual_microenv
```

Use this strict unique-architecture summary command:

```bash
python - <<'PY'
import csv
import glob
import os
from collections import defaultdict

patterns = [
    "checkpoints_pair_lr1e5_epoch50_level*_fold*/val_epoch_losses.csv",
    "checkpoints_pair_lr1e5_level5_tokenized_dit_fold*/val_epoch_losses.csv",
    "checkpoints_pair_lr1e5_level5_stochastic_dit_fold*/val_epoch_losses.csv",
    "checkpoints_pair_lr1e5_level5_cross_attention_dit_fold*/val_epoch_losses.csv",
    "checkpoints_pair_lr1e5_level5_hierarchical_dit_fold*/val_epoch_losses.csv",
    "checkpoints_pair_lr1e5_level5_consistency_dit_fold*/val_epoch_losses.csv",
    "checkpoints_pair_lr1e5_level5_rectified_dit_fold*/val_epoch_losses.csv",
]

paths = []
for pattern in patterns:
    paths.extend(glob.glob(pattern))

runs = []
for path in sorted(set(paths)):
    folder = os.path.dirname(path)
    with open(path, newline="") as f:
        data = list(csv.DictReader(f))
    if not data:
        continue

    last = data[-1]
    final_epoch = int(float(last["epoch"]))
    if final_epoch != 49:
        print(f"SKIP not 50 epochs: {folder}, final_epoch={final_epoch}")
        continue

    best = min(data, key=lambda row: float(row["loss_pair"]))
    level = last.get("level", "")
    variant = last.get("variant", "") or "default"
    fold = last.get("fold_index", "")
    arch = f"level{level}" if variant == "default" else f"level{level}_{variant}"
    canonical_arch = "level5" if arch == "level5_stochastic_dit" else arch

    runs.append({
        "arch": canonical_arch,
        "raw_arch": arch,
        "fold": fold,
        "folder": folder,
        "final_val": float(last["loss_pair"]),
        "best_val": float(best["loss_pair"]),
        "best_epoch": int(float(best["epoch"])),
    })

groups = defaultdict(list)
for run in runs:
    groups[(run["arch"], run["fold"])].append(run)

fold_best = []
for (arch, fold), items in groups.items():
    best_item = min(items, key=lambda item: item["best_val"])
    fold_best.append(best_item)

arch_groups = defaultdict(list)
for run in fold_best:
    arch_groups[run["arch"]].append(run)

summary = []
for arch, items in arch_groups.items():
    mean_best = sum(item["best_val"] for item in items) / len(items)
    mean_final = sum(item["final_val"] for item in items) / len(items)
    summary.append((mean_best, mean_final, arch, len(items), items))

summary.sort(key=lambda row: row[0])

print("rank,unique_architecture,n_folds,mean_best_val_loss_pair,mean_final_val_loss_pair")
for rank, (mean_best, mean_final, arch, n_folds, items) in enumerate(summary, 1):
    print(f"{rank},{arch},{n_folds},{mean_best:.8g},{mean_final:.8g}")

print("\nTOP 5 UNIQUE ARCHITECTURES, 50 EPOCH ONLY:")
for rank, (mean_best, mean_final, arch, n_folds, items) in enumerate(summary[:5], 1):
    print(f"{rank}. {arch}, folds={n_folds}, mean_best={mean_best:.8g}, mean_final={mean_final:.8g}")

print("\nWARNING CHECK:")
for mean_best, mean_final, arch, n_folds, items in summary:
    if n_folds != 5:
        print(f"{arch} has {n_folds} folds, expected 5")
PY
```

## TACC CSV Archive Command

Archive only CSV/config files for local analysis. Exclude `.pt` checkpoints.

```bash
tar -czf epoch50_model_selection_csv_results.tar.gz \
  checkpoints_pair_lr1e5_epoch50_level*_fold*/train_epoch_losses.csv \
  checkpoints_pair_lr1e5_epoch50_level*_fold*/val_epoch_losses.csv \
  checkpoints_pair_lr1e5_epoch50_level*_fold*/run_config.json \
  checkpoints_pair_lr1e5_level5_tokenized_dit_fold*/train_epoch_losses.csv \
  checkpoints_pair_lr1e5_level5_tokenized_dit_fold*/val_epoch_losses.csv \
  checkpoints_pair_lr1e5_level5_tokenized_dit_fold*/run_config.json \
  checkpoints_pair_lr1e5_level5_stochastic_dit_fold*/train_epoch_losses.csv \
  checkpoints_pair_lr1e5_level5_stochastic_dit_fold*/val_epoch_losses.csv \
  checkpoints_pair_lr1e5_level5_stochastic_dit_fold*/run_config.json \
  checkpoints_pair_lr1e5_level5_cross_attention_dit_fold*/train_epoch_losses.csv \
  checkpoints_pair_lr1e5_level5_cross_attention_dit_fold*/val_epoch_losses.csv \
  checkpoints_pair_lr1e5_level5_cross_attention_dit_fold*/run_config.json \
  checkpoints_pair_lr1e5_level5_hierarchical_dit_fold*/train_epoch_losses.csv \
  checkpoints_pair_lr1e5_level5_hierarchical_dit_fold*/val_epoch_losses.csv \
  checkpoints_pair_lr1e5_level5_hierarchical_dit_fold*/run_config.json \
  checkpoints_pair_lr1e5_level5_consistency_dit_fold*/train_epoch_losses.csv \
  checkpoints_pair_lr1e5_level5_consistency_dit_fold*/val_epoch_losses.csv \
  checkpoints_pair_lr1e5_level5_consistency_dit_fold*/run_config.json \
  checkpoints_pair_lr1e5_level5_rectified_dit_fold*/train_epoch_losses.csv \
  checkpoints_pair_lr1e5_level5_rectified_dit_fold*/val_epoch_losses.csv \
  checkpoints_pair_lr1e5_level5_rectified_dit_fold*/run_config.json
```

Check archive:

```bash
ls -lh epoch50_model_selection_csv_results.tar.gz
tar -tzf epoch50_model_selection_csv_results.tar.gz | head
```

Recommended TACC command for only Level 5 variants:

```bash
for variant in tokenized_dit stochastic_dit cross_attention_dit hierarchical_dit consistency_dit rectified_dit; do
  for fold in 0 1 2 3 4; do
    python train_five_level_world_model.py \
      --data-dir paired_training_h5ad_50k \
      --level 5 \
      --variant "$variant" \
      --batch-size 2048 \
      --epochs 50 \
      --max-steps -1 \
      --num-folds 5 \
      --fold-index "$fold" \
      --lr 1e-5 \
      --checkpoint-dir "checkpoints_pair_lr1e5_level5_${variant}_fold${fold}"
  done
done
```

## Current Selection

Do not finalize this section until the strict 50-epoch fold-average summary is run.

Previous best model level:

```text
Level 4
```

Backup:

```text
Level 3
```

Previous weak model in the first pass:

```text
Level 5
```

## Important Caveat

The current code uses a temporary internal encoder/decoder, not frozen scVI yet.

The next serious version should replace the temporary encoder with:

```text
frozen scVI encoder, latent_dim = 128
```

and train:

```text
z_source -> world model -> z_pred
z_target = scVI(y_target)
loss_pair = weighted MSE(z_pred, z_target)
```
