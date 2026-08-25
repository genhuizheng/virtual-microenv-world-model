# Future Record 01: Fixed scVI Latent Pipeline

## Goal

Replace the temporary internal expression autoencoder with a frozen scVI latent representation.

The intended training flow is:

```text
x_source -> frozen scVI encoder -> z_source
y_target -> frozen scVI encoder -> z_target
z_source, delta -> world model -> z_pred
loss(z_pred, z_target)
```

## Recommended Defaults

```text
scVI latent dimension: 128
scVI state during world-model training: frozen
world-model first tests: Level 2 and Level 3
main loss: Chreode-style latent loss
optional expression decoding: evaluation only at first
```

## Implementation Steps

1. Train or load a scVI model on the unified gene vocabulary.
2. Freeze the scVI encoder and decoder.
3. Cache source and target latents for the paired h5ad rows.
4. Train the world model directly in latent space.
5. Save latent-space checkpoints and epoch-level loss curves.
6. Add optional scVI decoder evaluation after the transition model is stable.

## Why This Matters

The current runnable model is useful for screening, but its internal autoencoder is not the final biological latent space.

The fixed scVI version should be treated as the first serious world-model implementation.
