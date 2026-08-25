# Two-stage Chreode/CellFlow training

This pipeline follows the manuscript representation contract:

1. H5AD matrices remain raw counts.
2. The loader applies `normalize_total(10^4)` followed by `log1p` in memory.
3. Stage 1 trains a Gaussian VAE and selects its checkpoint on held-out patients.
4. Stage 2 loads and freezes the VAE, then trains CellFlow in the 128-dimensional latent.
5. MOSCOT probability is used once as a loss weight. Use the `uniform_fast` pair dataset; do not combine probability-weighted resampling with probability-weighted training.

## Required TACC files

Upload these files to `/scratch/10119/ghzheng/Virtual_microenv`:

- `train_stage1_gaussian_vae.py`
- `train_five_level_world_model.py`
- `five_level_cell_world_model.py`
- `sample_paired_h5ad_dataloader.py`
- `chreode_loss.py`
- `TACC_1000k_full_cellflow_level6.sh`

The raw-count pair directory must be:

```text
paired_training_h5ad_1000k_fast/
```

## Test run

```bash
sbatch TACC_1000k_full_cellflow_level6.sh test
```

Test mode runs five Stage-1 training batches, two Stage-1 validation batches, and ten Stage-2 steps.

## Full run

```bash
sbatch TACC_1000k_full_cellflow_level6.sh full
```

Full-mode outputs:

```text
checkpoints_stage1_gaussian_vae_1000k_fold0/
checkpoints_1000k_paper_vae_cellflow_level6_fold0/
```

The Stage-2 validation checkpoint is:

```text
checkpoints_1000k_paper_vae_cellflow_level6_fold0/best_model_level6_fold0.pt
```

The patient split is determined from `paired_moscot_pairs.tsv` using `patient_id`; shards are storage only and no longer define folds.
