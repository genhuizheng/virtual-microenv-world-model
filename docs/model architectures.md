# Model Architectures

## Core Decision

Use **scVI as the VAE / latent cell-state encoder** for the real world model.

scVI is now implemented as a real Stage-1 representation (`train_stage1_scvi.py`,
`scvi_stage1_representation.py`, `--representation-type scvi` /
`--stage1-scvi-checkpoint` in `train_five_level_world_model.py`), trained with
scvi-tools' native NB/ZINB reconstruction likelihood on raw counts rather than
the ad hoc `GaussianExpressionVAE`. It replaces the "Level 0: Planned real
encoder" row below for any new screening run. See
`docs/stage1_scvi_notes.md` for the exact encode()/decode() contract and the
design choices made (no batch-key conditioning; `decode()` returns
`log1p(px_scale * 1e4)`, the same units as `log1p_10k` elsewhere in this
project). `--expression-transform none` (raw counts) is required with a scVI
checkpoint; `log1p_10k` remains required for the Gaussian VAE checkpoint.

The world-model levels should all operate on the same latent representation:

```text
expression x_t -> scVI encoder -> z_t
z_t, delta -> world model -> z_{t+delta}
z_{t+delta} -> scVI decoder or latent-space evaluation
```

This is better than training a small custom expression autoencoder inside every transition model because scVI is already designed for single-cell count data, batch effects, library-size variation, and stable low-dimensional biological latents.

## What Is Implemented Now

The current Python implementation is a **first testing scaffold**, not the final full architecture suite.

Implemented now:

```text
five_level_cell_world_model.py
chreode_loss.py
train_five_level_world_model.py
```

The code includes runnable prototype classes for Levels 1-6. The first screening pass selected Level 4, then compared Level 4 against the expanded Level 5 variant family. Level 6 adds a CellFlow-inspired flow-matching velocity-field baseline.

```text
Level 4: Waddington / Chreode-style residual DiT
Level 5: tokenized, stochastic, cross-attention, hierarchical, consistency, and rectified variants
Level 6: CellFlow-inspired flow-matching velocity field
```

The current code also includes a small internal PyTorch expression autoencoder only so the paired h5ad dataloader can be tested immediately:

```text
batch["x"] -> internal encoder -> z_t -> transition -> z_pred -> internal decoder -> batch["y"]
```

For the real TACC version, replace this internal autoencoder with pretrained scVI latent extraction.

## Current Dataloader Contract

The paired h5ad dataloader returns:

```python
batch["x"]       # source / baseline expression, [B, G]
batch["y"]       # target / posttreatment expression, [B, G]
batch["weight"]  # MOSCOT probability weight, [B]
batch["delta"]   # elapsed time, [B]
```

If the h5ad metadata has a real elapsed-time column, use:

```bash
--time-delta-col your_column_name
```

Otherwise, `delta = 1.0` is used as a one-step default.

## Architecture Roadmap

| Level | Architecture | Current status | Purpose |
|---:|---|---|---|
| 0 | scVI VAE latent encoder | Planned real encoder | Shared latent cell-state space |
| 1 | Transformer residual model | Prototype implemented | Basic attention residual baseline |
| 2 | DiT residual model | Implemented for first DiT testing | First main DiT test |
| 3 | Gated one-step residual DiT | Implemented for first DiT testing | Adds explicit elapsed-time gate |
| 4 | Waddington / Chreode-style residual DiT | Prototype only | Biological structured residual |
| 5 | Advanced DiT variants | Implemented screening variants | Scaling and stronger variants |
| 6 | CellFlow-inspired velocity field | Implemented screening variant | Flow matching on interpolated source-target latent states |

## Level 0: scVI Latent Space

Recommended representation:

```text
z_t = scVI_encoder(x_t)
z_{t+delta} = scVI_encoder(x_{t+delta})
```

Training target can be latent-space first:

```text
loss_z = || z_pred - z_target ||^2
```

Expression reconstruction can be added later:

```text
y_pred = scVI_decoder(z_pred)
loss_x = expression reconstruction loss
```

Recommendation:

```text
Pretrain or load scVI first, freeze it for initial transition-model testing,
then optionally fine-tune later.
```

## Level 1: Transformer Residual Model

Form:

```text
z_pred = z_t + R_theta(z_t, delta)
```

Role:

```text
Simple attention-based residual baseline.
```

Status:

```text
Prototype implemented.
```

## Level 2: DiT Residual Model

Form:

```text
z_pred = z_t + DiT_theta(z_t, delta)
```

Uses:

```text
latent tokenization
Fourier time embedding
AdaLN-Zero-style time conditioning
Transformer self-attention blocks
```

Role:

```text
This is the first DiT model to test.
```

Status:

```text
Implemented for first testing.
```

## Level 3: Gated One-Step Residual DiT

Form:

```text
z_pred = z_t + alpha(delta) * R_theta(z_t, delta)
alpha(delta) = 1 - exp(-delta / tau)
```

Role:

```text
Best first biological-temporal test because it enforces smaller changes for shorter elapsed time.
```

Status:

```text
Implemented for first testing.
```

## Level 4: Waddington / Chreode-Style Residual DiT

Intended form:

```text
R_theta =
  -grad_z U_theta(z_t, delta)
  + S_theta(z_t, delta) z_t
  + sigma_theta(z_t, delta) * epsilon
```

Meaning:

```text
-grad_z U_theta         downhill potential flow
S_theta z_t             antisymmetric rotational flow
sigma_theta * epsilon   stochastic biological spread
```

Status:

```text
Prototype only.
```

Important caution:

```text
This level needs careful validation. The current code captures the intended decomposition,
but it should not be treated as a final Chreode implementation yet.
```

## Level 5: Advanced DiT Variants

Level 5 is not one single random model. It is a family of possible stronger DiT variants.

The current code now exposes several runnable screening variants through:

```text
--level 5 --variant <name>
```

Current implemented variants:

| Label | Variant name | Include now? | Notes |
|---|---|---|---|
| 5B | `tokenized_dit` | Yes | Deterministic tokenized latent AdaLN-Zero DiT |
| 5C | `stochastic_dit` | Yes | Tokenized AdaLN-Zero DiT with stochastic residual head |
| 5D | `cross_attention_dit` | Yes, prototype | Learned null context token until intervention/action tokens are added |
| 5E | `hierarchical_dit` | Yes, prototype | Hierarchical/U-Net-inspired token DiT |
| 5F | `consistency_dit` | Yes, prototype | One-step consistency-style direct/residual mixer |
| 5G | `rectified_dit` | Yes, prototype | Rectified-flow-inspired displacement predictor |
| 5H | `diffusion_denoising_dit` | No for now | Excluded because iterative denoising conflicts with one-step prediction |

For `stochastic_dit`, training samples:

```text
residual = mean + sigma * epsilon
```

During evaluation it uses:

```text
epsilon = 0
residual = mean
```

So it is stochastic, but not random in the sense of an uncontrolled architecture.

## Level 6: CellFlow-Inspired Flow Matching

Level 6 is a CellFlow-inspired velocity-field model.

Unlike Levels 1-5, it does not train primarily by directly predicting `z_target`.
Instead, it samples an interpolated latent state between paired source and target cells:

```text
t ~ Uniform(0, 1)
z_t = (1 - t) z_source + t z_target
target_velocity = z_target - z_source
pred_velocity = v_theta(z_t, t, delta)
loss_cellflow = weighted MSE(pred_velocity, target_velocity)
```

In the current MOSCOT-paired setup, the MOSCOT pairs act as the source-target coupling that CellFlow normally obtains through OT/UOT coupling.

Run it with:

```text
--level 6 --loss-mode cellflow
```

The script still logs `loss_pair`, `metric_mmd`, `metric_w2`, and `metric_drift` using the one-step prediction from the learned velocity field, so Level 6 can be compared against Levels 1-5.

## Recommended Testing Order

First test with the current paired h5ad dataloader:

```text
Level 2: DiT residual model
Level 3: gated one-step residual DiT
```

Then compare against:

```text
Level 1: Transformer residual baseline
```

Only after Level 2/3 behave sensibly:

```text
Level 4: Waddington / Chreode-style prototype
Level 5: stochastic or cross-attention variants
Level 6: CellFlow-inspired flow-matching velocity field
```

## First-Round Testing Loss

The training script uses MOSCOT-paired latent MSE by default for first-round screening:

```text
loss_mode = pair_mse
L = weighted MSE(z_pred, z_target)
```

Secondary comparison metrics are logged every run:

```text
metric_mmd
metric_w2
metric_drift
metric_down
```

The paper-style Chreode objective is still available as an explicit option:

```text
--loss-mode chreode
L = lambda_mmd L_mmd + lambda_w2 L_w2 + lambda_drift L_drift + lambda_down L_down
```

The CellFlow-inspired objective is available as:

```text
--level 6 --loss-mode cellflow
L = weighted MSE(v_theta(z_t, t, delta), z_target - z_source)
```

Paper settings to remember:

```text
scVI latent dimension = 128
paper W-DiT stochastic samples K = 8 during pretraining
paper stage-2 batch size = 512 source cells per step
paper optimizer = AdamW, weight decay 0.01, gradient clip 1.0
```

Current first-round test simplification:

```text
Level 2/3 DiT tests use one prediction per source cell, effectively K = 1.
This is intentional for checking whether the loss decreases before implementing full stochastic W-DiT training.
```

Implemented components:

```text
L_mmd      multi-bandwidth RBF MMD between predicted and target latent batches
L_w2       entropic Sinkhorn W2 between predicted and target latent batches
L_drift    paired displacement consistency adapted for MOSCOT pairs
L_down     downhill regularizer, active only when potential_flow is available
```

Important adaptation:

```text
The paper trains on unpaired population snapshots.
This project currently has MOSCOT paired rows.
Therefore L_drift is implemented as paired displacement consistency:

    z_pred - z_source should match z_target - z_source
```

For Level 2 and Level 3, `L_down = 0` because these models do not have a potential-flow branch. That is expected.

The output loss files are:

```text
checkpoints_level{level}/step_losses.csv
checkpoints_level{level}/train_epoch_losses.csv
checkpoints_level{level}/val_epoch_losses.csv
```

If `--checkpoint-dir` is provided, that custom directory is used instead.

Checkpoint files are level-marked:

```text
world_model_level{level}_fold{fold}_step{step}.pt
best_model_level{level}_fold{fold}.pt
```

For Level 5 variants, checkpoint names also include the variant:

```text
world_model_level5_{variant}_fold{fold}_step{step}.pt
best_model_level5_{variant}_fold{fold}.pt
```

The columns are:

```text
loss
level
loss_mode
loss_mmd
loss_w2
loss_drift
loss_down
loss_expr
loss_latent_aux
loss_source_recon
```

For first-round model selection, the main expectation is simple:

```text
epoch mean loss should decrease across training.
loss_mmd, loss_w2, and loss_drift should generally trend down.
```

## Recommended First Commands

First DiT test:

```bash
python train_five_level_world_model.py \
  --data-dir paired_training_h5ad_50k \
  --level 2 \
  --batch-size 64 \
  --epochs 5 \
  --max-steps -1
```

Gated DiT test:

```bash
python train_five_level_world_model.py \
  --data-dir paired_training_h5ad_50k \
  --level 3 \
  --batch-size 64 \
  --epochs 5 \
  --max-steps -1
```

Quick smoke test before a full epoch:

```bash
python train_five_level_world_model.py \
  --data-dir paired_training_h5ad_50k \
  --level 2 \
  --batch-size 32 \
  --max-steps 20 \
  --sinkhorn-iters 20 \
  --checkpoint-dir smoke_level2
```

Use the smoke test only to check that the training loop runs. Use the full-epoch runs to judge whether the loss is truly going down.

## Next Implementation Step

The next important implementation step is to add a scVI latent pipeline:

```text
1. Train or load scVI on the unified gene expression data.
2. Encode source expression x into z_source.
3. Encode target expression y into z_target.
4. Train Level 2/3 transition model in latent space.
5. Keep expression-space decoding optional for later evaluation.
```

This will make the implementation match the intended foundation-style cell world model more closely than the temporary internal autoencoder.
