# 500k Training Loss Function Design

## Goal

Design and screen Chreode-style loss functions on the 500k MOSCOT-paired dataset.

The architecture selection step is already complete. The next step is not to search more model structures, but to tune the training objective.

## Chosen Architecture for Loss Search

Primary architecture:

```text
level5_hierarchical_dit
```

Reason:

```text
It was the strongest DiT-style model in the 500k architecture screen.
It is more expressive than the simple level1 baseline.
It is easier to interpret for DiT-family loss selection than level6_cellflow, because level6 uses a different flow-matching objective.
```

Reference 500k result:

| Architecture | Mean best pair | Mean final pair | Mean final MMD | Mean final W2 |
|---|---:|---:|---:|---:|
| `level5_hierarchical_dit` | 3.381466e-05 | 3.386767e-05 | 0.074395 | 0.007283 |

## Current Chreode Loss Components

The current code implements:

```text
L_chreode =
  lambda_mmd   * L_mmd
+ lambda_w2    * L_w2
+ lambda_drift * L_drift
+ lambda_down  * L_down
```

where:

```text
L_mmd:
Population-level RBF MMD between predicted latent population and target latent population.

L_w2:
Population-level Sinkhorn/Wasserstein transport cost between predicted and target latent populations.

L_drift:
Paired latent displacement consistency.
For the current MOSCOT-paired setting:
L_drift = MSE((z_pred - z_source), (z_target - z_source))
        = MSE(z_pred, z_target)
So this is effectively the same anchor as loss_pair.

L_down:
Downhill potential-flow regularizer.
This is only active when the model returns `potential_flow`, mainly the Level 4 prototype.
For Level 5 hierarchical DiT, this term is effectively zero.
```

## Important Scaling Issue

From the 500k model ranking:

```text
pair loss scale: about 1e-05 to 1e-04
MMD scale:       about 1e-02 to 1e-01
W2 scale:        about 1e-03 to 1e-02
```

Therefore, using:

```text
lambda_mmd = 1
lambda_w2 = 1
lambda_drift = 1
```

would make MMD/W2 dominate the loss. That is not a fair loss-function screen.

The loss search should keep `lambda_drift=1.0` as the pair anchor, and test small population weights.

## Updated Screening Strategy

Use one architecture and one fold for the first loss-parameter screen:

```text
architecture: level5_hierarchical_dit
dataset: paired_training_h5ad_500k_fast
screen fold: fold 0
epochs: 50
learning rate: 1e-5
batch size: 2048
loss mode: chreode
```

After the fold-0 screen, select the best 2-3 loss settings and confirm them across all five folds.

This is more efficient than immediately running every loss setting on every fold.

## Proposed Loss Search Grid

Primary selection metric:

```text
validation loss_pair
```

Secondary metrics:

```text
metric_mmd
metric_w2
metric_drift
```

The best loss should reduce or preserve pair loss while improving MMD/W2.

| Loss tag | lambda_drift | lambda_mmd | lambda_w2 | lambda_down | Interpretation |
|---|---:|---:|---:|---:|---|
| `pair_anchor` | 1.0 | 0.0 | 0.0 | 0.0 | Control; equivalent to pair/drift-only Chreode |
| `weak_pop` | 1.0 | 1e-4 | 1e-3 | 0.0 | Recommended first Chreode-style loss |
| `medium_pop` | 1.0 | 3e-4 | 3e-3 | 0.0 | Stronger population regularization |
| `strong_pop` | 1.0 | 1e-3 | 1e-2 | 0.0 | Stress test; may hurt pair prediction |
| `mmd_only_weak` | 1.0 | 1e-4 | 0.0 | 0.0 | Tests whether MMD helps alone |
| `w2_only_weak` | 1.0 | 0.0 | 1e-3 | 0.0 | Tests whether W2 helps alone |

The executable fold-0 screen uses a broader grid:

| Loss tag | lambda_drift | lambda_mmd | lambda_w2 | lambda_down |
|---|---:|---:|---:|---:|
| `pair_anchor` | 1.0 | 0.0 | 0.0 | 0.0 |
| `mmd_1e6` | 1.0 | 1e-6 | 0.0 | 0.0 |
| `mmd_3e6` | 1.0 | 3e-6 | 0.0 | 0.0 |
| `mmd_1e5` | 1.0 | 1e-5 | 0.0 | 0.0 |
| `mmd_3e5` | 1.0 | 3e-5 | 0.0 | 0.0 |
| `mmd_1e4` | 1.0 | 1e-4 | 0.0 | 0.0 |
| `mmd_3e4` | 1.0 | 3e-4 | 0.0 | 0.0 |
| `mmd_1e3` | 1.0 | 1e-3 | 0.0 | 0.0 |
| `mmd_3e3` | 1.0 | 3e-3 | 0.0 | 0.0 |
| `w2_1e5` | 1.0 | 0.0 | 1e-5 | 0.0 |
| `w2_3e5` | 1.0 | 0.0 | 3e-5 | 0.0 |
| `w2_1e4` | 1.0 | 0.0 | 1e-4 | 0.0 |
| `w2_3e4` | 1.0 | 0.0 | 3e-4 | 0.0 |
| `w2_1e3` | 1.0 | 0.0 | 1e-3 | 0.0 |
| `w2_3e3` | 1.0 | 0.0 | 3e-3 | 0.0 |
| `w2_1e2` | 1.0 | 0.0 | 1e-2 | 0.0 |
| `w2_3e2` | 1.0 | 0.0 | 3e-2 | 0.0 |
| `pop_1e6_1e5` | 1.0 | 1e-6 | 1e-5 | 0.0 |
| `pop_3e6_3e5` | 1.0 | 3e-6 | 3e-5 | 0.0 |
| `pop_1e5_1e4` | 1.0 | 1e-5 | 1e-4 | 0.0 |
| `pop_3e5_3e4` | 1.0 | 3e-5 | 3e-4 | 0.0 |
| `pop_1e4_1e3` | 1.0 | 1e-4 | 1e-3 | 0.0 |
| `pop_3e4_3e3` | 1.0 | 3e-4 | 3e-3 | 0.0 |
| `pop_1e3_1e2` | 1.0 | 1e-3 | 1e-2 | 0.0 |
| `pop_3e3_3e2` | 1.0 | 3e-3 | 3e-2 | 0.0 |
| `pop_mmd_low_w2_mid` | 1.0 | 1e-5 | 1e-3 | 0.0 |
| `pop_mmd_mid_w2_low` | 1.0 | 1e-4 | 1e-4 | 0.0 |
| `pop_mmd_mid_w2_high` | 1.0 | 1e-4 | 1e-2 | 0.0 |
| `pop_mmd_high_w2_mid` | 1.0 | 1e-3 | 1e-3 | 0.0 |
| `drift_03_pair_anchor` | 0.3 | 0.0 | 0.0 | 0.0 |
| `drift_3_pair_anchor` | 3.0 | 0.0 | 0.0 | 0.0 |
| `drift_03_weak_pop` | 0.3 | 1e-4 | 1e-3 | 0.0 |
| `drift_3_weak_pop` | 3.0 | 1e-4 | 1e-3 | 0.0 |
| `drift_03_medium_pop` | 0.3 | 3e-4 | 3e-3 | 0.0 |
| `drift_3_medium_pop` | 3.0 | 3e-4 | 3e-3 | 0.0 |

## Recommended First Decision Rule

Choose the loss setting that satisfies:

```text
1. mean_best_pair is close to or better than pair-only baseline
2. mean_final_pair is stable and does not drift upward
3. mean_final_mmd improves meaningfully
4. mean_final_w2 improves or remains close
```

For this stage, avoid selecting a loss that improves MMD but destroys pair loss.

## TACC Commands

Run these on:

```text
/scratch/10119/ghzheng/Virtual_microenv
```

### Recommended Fold-0 Screen

Upload these files to TACC:

```text
train_five_level_world_model.py
tacc_commands/TACC_thirdroundselection_lossfunction_500k_fold0_all.sh
tacc_commands/TACC_thirdroundselection_lossfunction_500k_fold0_mmd.sh
tacc_commands/TACC_thirdroundselection_lossfunction_500k_fold0_w2.sh
tacc_commands/TACC_thirdroundselection_lossfunction_500k_fold0_pop.sh
tacc_commands/TACC_thirdroundselection_lossfunction_500k_fold0_drift.sh
rank_500k_loss_screen.py
```

The all-in-one script is:

```bash
sbatch tacc_commands/TACC_thirdroundselection_lossfunction_500k_fold0_all.sh
```

For safer shorter jobs, submit the split scripts instead:

```bash
sbatch tacc_commands/TACC_thirdroundselection_lossfunction_500k_fold0_mmd.sh
sbatch tacc_commands/TACC_thirdroundselection_lossfunction_500k_fold0_w2.sh
sbatch tacc_commands/TACC_thirdroundselection_lossfunction_500k_fold0_pop.sh
sbatch tacc_commands/TACC_thirdroundselection_lossfunction_500k_fold0_drift.sh
```

Each script writes or refreshes:

```text
500k_loss_screen_fold0_summary.csv
```

`pair_anchor` is included in each split script as a shared control. It writes to the same checkpoint folder each time. If you want to avoid rerunning it, submit the MMD script first, then remove or comment `pair_anchor` in the other split scripts before submitting them.

### Manual Level 5 Hierarchical DiT Loss Search

```bash
source /home1/10119/ghzheng/.bashrc
conda activate worldmodeltraining

cd /scratch/10119/ghzheng/Virtual_microenv

for fold in 0 1 2 3 4; do
  echo "===== loss=pair_anchor fold=$fold ====="
  python train_five_level_world_model.py \
    --data-dir paired_training_h5ad_500k_fast \
    --level 5 \
    --variant hierarchical_dit \
    --loss-mode chreode \
    --lambda-drift 1.0 \
    --lambda-mmd 0.0 \
    --lambda-w2 0.0 \
    --lambda-down 0.0 \
    --batch-size 2048 \
    --epochs 50 \
    --max-steps -1 \
    --num-folds 5 \
    --fold-index "$fold" \
    --lr 1e-5 \
    --save-every 0 \
    --no-save-each-epoch \
    --sinkhorn-iters 20 \
    --checkpoint-dir "checkpoints_500k_loss_level5_hierarchical_dit_pair_anchor_fold${fold}"
done

for fold in 0 1 2 3 4; do
  echo "===== loss=weak_pop fold=$fold ====="
  python train_five_level_world_model.py \
    --data-dir paired_training_h5ad_500k_fast \
    --level 5 \
    --variant hierarchical_dit \
    --loss-mode chreode \
    --lambda-drift 1.0 \
    --lambda-mmd 1e-4 \
    --lambda-w2 1e-3 \
    --lambda-down 0.0 \
    --batch-size 2048 \
    --epochs 50 \
    --max-steps -1 \
    --num-folds 5 \
    --fold-index "$fold" \
    --lr 1e-5 \
    --save-every 0 \
    --no-save-each-epoch \
    --sinkhorn-iters 20 \
    --checkpoint-dir "checkpoints_500k_loss_level5_hierarchical_dit_weak_pop_fold${fold}"
done

for fold in 0 1 2 3 4; do
  echo "===== loss=medium_pop fold=$fold ====="
  python train_five_level_world_model.py \
    --data-dir paired_training_h5ad_500k_fast \
    --level 5 \
    --variant hierarchical_dit \
    --loss-mode chreode \
    --lambda-drift 1.0 \
    --lambda-mmd 3e-4 \
    --lambda-w2 3e-3 \
    --lambda-down 0.0 \
    --batch-size 2048 \
    --epochs 50 \
    --max-steps -1 \
    --num-folds 5 \
    --fold-index "$fold" \
    --lr 1e-5 \
    --save-every 0 \
    --no-save-each-epoch \
    --sinkhorn-iters 20 \
    --checkpoint-dir "checkpoints_500k_loss_level5_hierarchical_dit_medium_pop_fold${fold}"
done

for fold in 0 1 2 3 4; do
  echo "===== loss=strong_pop fold=$fold ====="
  python train_five_level_world_model.py \
    --data-dir paired_training_h5ad_500k_fast \
    --level 5 \
    --variant hierarchical_dit \
    --loss-mode chreode \
    --lambda-drift 1.0 \
    --lambda-mmd 1e-3 \
    --lambda-w2 1e-2 \
    --lambda-down 0.0 \
    --batch-size 2048 \
    --epochs 50 \
    --max-steps -1 \
    --num-folds 5 \
    --fold-index "$fold" \
    --lr 1e-5 \
    --save-every 0 \
    --no-save-each-epoch \
    --sinkhorn-iters 20 \
    --checkpoint-dir "checkpoints_500k_loss_level5_hierarchical_dit_strong_pop_fold${fold}"
done

for fold in 0 1 2 3 4; do
  echo "===== loss=mmd_only_weak fold=$fold ====="
  python train_five_level_world_model.py \
    --data-dir paired_training_h5ad_500k_fast \
    --level 5 \
    --variant hierarchical_dit \
    --loss-mode chreode \
    --lambda-drift 1.0 \
    --lambda-mmd 1e-4 \
    --lambda-w2 0.0 \
    --lambda-down 0.0 \
    --batch-size 2048 \
    --epochs 50 \
    --max-steps -1 \
    --num-folds 5 \
    --fold-index "$fold" \
    --lr 1e-5 \
    --save-every 0 \
    --no-save-each-epoch \
    --sinkhorn-iters 20 \
    --checkpoint-dir "checkpoints_500k_loss_level5_hierarchical_dit_mmd_only_weak_fold${fold}"
done

for fold in 0 1 2 3 4; do
  echo "===== loss=w2_only_weak fold=$fold ====="
  python train_five_level_world_model.py \
    --data-dir paired_training_h5ad_500k_fast \
    --level 5 \
    --variant hierarchical_dit \
    --loss-mode chreode \
    --lambda-drift 1.0 \
    --lambda-mmd 0.0 \
    --lambda-w2 1e-3 \
    --lambda-down 0.0 \
    --batch-size 2048 \
    --epochs 50 \
    --max-steps -1 \
    --num-folds 5 \
    --fold-index "$fold" \
    --lr 1e-5 \
    --save-every 0 \
    --no-save-each-epoch \
    --sinkhorn-iters 20 \
    --checkpoint-dir "checkpoints_500k_loss_level5_hierarchical_dit_w2_only_weak_fold${fold}"
done
```

## Ranking Script

```bash
python - <<'PY'
import csv
import glob
import math
import os
from collections import defaultdict

paths = glob.glob("checkpoints_500k_loss_level5_hierarchical_dit_*_fold*/val_epoch_losses.csv")

def value(row, name):
    if name not in row or row[name] in ("", "nan", "NaN"):
        return math.nan
    return float(row[name])

runs = []
for path in sorted(paths):
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

    tag = folder.replace("checkpoints_500k_loss_level5_hierarchical_dit_", "")
    tag = tag.rsplit("_fold", 1)[0]
    fold = last["fold_index"]
    best = min(value(row, "loss_pair") for row in data)

    runs.append({
        "tag": tag,
        "fold": fold,
        "best_pair": best,
        "final_pair": value(last, "loss_pair"),
        "final_loss": value(last, "loss"),
        "final_mmd": value(last, "metric_mmd"),
        "final_w2": value(last, "metric_w2"),
        "final_drift": value(last, "metric_drift"),
        "final_down": value(last, "metric_down"),
    })

groups = defaultdict(list)
for run in runs:
    groups[run["tag"]].append(run)

summary = []
for tag, rows in groups.items():
    if len({row["fold"] for row in rows}) != 5:
        print(f"WARNING: {tag} has {len({row['fold'] for row in rows})} folds, expected 5")
    mean = lambda key: sum(row[key] for row in rows) / len(rows)
    summary.append((
        tag,
        len(rows),
        mean("best_pair"),
        mean("final_pair"),
        mean("final_loss"),
        mean("final_mmd"),
        mean("final_w2"),
        mean("final_drift"),
        mean("final_down"),
    ))

summary.sort(key=lambda row: (row[2], row[5], row[6]))

print("rank,loss_tag,n_folds,mean_best_pair,mean_final_pair,mean_final_loss,mean_final_mmd,mean_final_w2,mean_final_drift,mean_final_down")
for i, row in enumerate(summary, 1):
    print(i, *row, sep=",")
PY
```

## Expected Outcome

The most likely useful setting is:

```text
weak_pop:
lambda_drift = 1.0
lambda_mmd = 1e-4
lambda_w2 = 1e-3
lambda_down = 0.0
```

This setting gives population structure a small but nonzero influence while preserving the MOSCOT-paired prediction objective.

If `weak_pop` improves MMD/W2 without hurting pair loss, it should become the preferred Level 5 hierarchical DiT loss.

If all population-weighted settings hurt pair loss, keep `pair_anchor` for Level 5 and use Level 6 CellFlow as the biologically motivated dynamics model.
