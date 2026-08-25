# 500k Model Selection Summary

## Goal

Rank the selected model architectures on the larger 500k MOSCOT-paired training dataset.

This is the second-round architecture selection after the 50k initial screening.

## Data

```text
dataset: paired_training_h5ad_500k_fast
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
epochs: 50
learning rate: 1e-5
batch size: 2048
selection metric: validation loss_pair
```

Each architecture was evaluated across:

```text
fold_index = 0, 1, 2, 3, 4
```

## Metric Definitions

```text
Mean best pair:
Average across 5 folds of the lowest validation pair loss reached during training.

Mean final pair:
Average across 5 folds of the validation pair loss at the final epoch.

Mean final MMD:
Average across 5 folds of the final validation population MMD metric.

Mean final W2:
Average across 5 folds of the final validation Sinkhorn/Wasserstein population metric.

Mean final drift:
Average across 5 folds of the final validation paired latent drift metric.

Mean final down:
Average across 5 folds of the final validation downhill regularization metric.

Mean final cellflow:
Average across 5 folds of the final validation flow-matching loss. This is only defined for Level 6.
```

## Training Objective

For Levels 1, 4, and 5 variants, the training objective is:

```text
loss = loss_pair
loss_pair = weighted MSE(z_pred, z_target)
```

For Level 6 CellFlow, the training objective is:

```text
t ~ Uniform(0, 1)
z_t = (1 - t) z_source + t z_target
loss_cellflow = weighted MSE(v_theta(z_t, t, delta), z_target - z_source)
```

For Level 6, `loss_pair` is still logged as a comparable downstream prediction metric.

## 500k Ranking

| Rank | Code name | Architecture name | Folds | Mean best pair | Mean final pair | Mean final MMD | Mean final W2 | Mean final drift | Mean final down | Mean final cellflow | Decision |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | `level6_cellflow` | Simplified CellFlow / flow-matching model | 5 | 1.288247e-05 | 1.313826e-05 | 0.047185 | 0.002359 | 1.313826e-05 | 0.000000 | 1.307107e-05 | Best overall |
| 2 | `level1` | Plain latent residual baseline | 5 | 1.384875e-05 | 1.589118e-05 | 0.120286 | 0.002747 | 1.589118e-05 | 0.000000 | N/A | Strong simple baseline |
| 3 | `level5_hierarchical_dit` | Hierarchical DiT variant | 5 | 3.381466e-05 | 3.386767e-05 | 0.074395 | 0.007283 | 3.386767e-05 | 0.000000 | N/A | Best DiT-style model |
| 4 | `level5` | Default stochastic AdaLN-Zero DiT | 5 | 4.643532e-05 | 4.853171e-05 | 0.097340 | 0.010355 | 4.853171e-05 | 0.000000 | N/A | Stable, useful comparator |
| 5 | `level5_cross_attention_dit` | Cross-attention DiT variant | 5 | 1.508708e-04 | 1.525000e-04 | 0.078637 | 0.025213 | 1.525000e-04 | 0.000000 | N/A | Keep as top-5 variant |
| 6 | `level4` | Waddington/Chreode-style residual DiT | 5 | 9.104884e-04 | 1.004541e-02 | 0.108776 | 0.922872 | 1.004541e-02 | 0.003877 | N/A | Drop from next round unless used as control |

## Top 5 Architectures

```text
1. level6_cellflow
2. level1
3. level5_hierarchical_dit
4. level5
5. level5_cross_attention_dit
```

## Interpretation

Level 6 CellFlow is the best-performing architecture on the 500k retest by validation pair loss, and it also has the lowest final W2 among the tested models. This suggests that the simplified flow-matching objective is a strong candidate for the next stage.

Level 1 remains a very strong baseline. Its pair loss is close to Level 6, although its final MMD is higher. This model should be retained as the simple baseline because it is easy to train and competitive.

Among the DiT-style models, the hierarchical DiT variant is the strongest. The default Level 5 AdaLN-Zero DiT is stable but weaker than hierarchical DiT. The cross-attention DiT remains in the top five but has a noticeably higher pair loss.

Level 4 performed poorly on 500k, especially in final pair loss and W2. It should not be prioritized for the next selection round unless it is needed as a comparison/control model.

## Recommended Next Step

Use the following architectures for the next round:

```text
level6_cellflow
level1
level5_hierarchical_dit
level5
level5_cross_attention_dit
```

The main ranking metric should remain validation `loss_pair`, with `metric_mmd` and `metric_w2` used as population-level sanity checks. For Level 6, also report `loss_cellflow` because it is the actual training objective.
