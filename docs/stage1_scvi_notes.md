# Stage-1 scVI Notes

Real scvi-tools replacement for the ad hoc `GaussianExpressionVAE` Stage 1.

## Files

```text
train_stage1_scvi.py            trains scVI, saves a lightweight checkpoint
scvi_stage1_representation.py   frozen encode()/decode() wrapper for Stage 2
```

`five_level_cell_world_model.py` gained `--representation-type scvi` and a
`scvi_checkpoint` config field; `train_five_level_world_model.py` gained
`--stage1-scvi-checkpoint` (mutually exclusive with `--stage1-vae-checkpoint`).

## Design choices

1. **No batch_key.** scVI is trained with a single dummy batch category
   (`n_batch=1`). Threading a real per-cell batch covariate (e.g.
   `dataset_id`) through `decode()` would require every call site that
   currently does `model.encode(x)` / `model.decode(z)` to also carry batch
   metadata for *predicted* latents that don't correspond to any real cell.
   Can be added later without changing the public `encode()`/`decode()`
   signatures used everywhere else in the pipeline.

2. **`decode(z)` returns `log1p(px_scale * 1e4)`**, not raw counts and not
   px_rate for an observed library. `px_scale` (scVI's library-size-independent
   normalized expression, sums to 1 per cell) is decoded against a *fixed*
   reference library size (`--reference-library-size`, default 1e4), matching
   the `log1p_10k` units used everywhere else in this project
   (`transform_expression_rows()` in `sample_paired_h5ad_dataloader.py`).
   This is the same trick `scvi.model.SCVI.get_normalized_expression()` uses
   internally.

3. **`encode(x)` requires raw counts** (`--expression-transform none`), not
   `log1p_10k`. scVI applies its own internal `log1p` before the encoder
   (`log_variational=True`) and its NB/ZINB likelihood is defined on counts.
   Feeding it `log1p_10k`-transformed data would double-transform.

4. **No `scvi.model.SCVI.save()`/`.load()`.** The checkpoint
   (`stage1_scvi_v1`) stores only the underlying `scvi.module.VAE`
   `state_dict()` plus its plain constructor kwargs, not a full
   `SCVI` model bundle with its AnnData registry. Stage 2 rebuilds the module
   directly (`VAE(**module_init_kwargs)` + `load_state_dict`) — no AnnData
   needs to travel with the checkpoint, and loading works identically
   whether Stage 1 and Stage 2 run on the same machine or not (e.g. Stage 1
   on one TACC node, Stage 2 on another).

5. **Patient-level fold split matches Stage 2 exactly.** Both
   `train_stage1_scvi.py` and `PairedH5ADBatchDataset` (used by
   `train_five_level_world_model.py`) call the same
   `compute_group_to_fold()` in `sample_paired_h5ad_dataloader.py`. Using the
   same `--group-column patient_id --num-folds N --fold-index K --seed S`
   for both stages guarantees the scVI encoder never saw the patients a
   transition model is later validated on.

## Commands

Stage 1 (one fold shown; repeat `--fold-index 0..4` for 5-fold coverage):

```bash
python train_stage1_scvi.py \
  --data-dir paired_training_h5ad_50k \
  --checkpoint-dir stage1_scvi_50k \
  --max-epochs 100 \
  --latent-dim 128 \
  --num-folds 5 --fold-index 0 \
  --group-column patient_id \
  --device cuda --overwrite
```

Stage 2, using the frozen scVI checkpoint from the matching fold:

```bash
python train_five_level_world_model.py \
  --data-dir paired_training_h5ad_50k \
  --level 2 \
  --stage1-scvi-checkpoint stage1_scvi_50k/best_stage1_scvi_fold0.pt \
  --expression-transform none \
  --group-column patient_id --num-folds 5 --fold-index 0 \
  --latent-dim 128
```

`tacc_commands/TACC_stage1_scvi.sh` wraps the Stage-1 command above in the
same sbatch style as the other `TACC_*` scripts (`-p gh`, `MCB26031`,
`worldmodeltraining` conda env).

## One-time setup required on TACC

`scvi-tools` is not yet in the `worldmodeltraining` conda env. Before
submitting `TACC_stage1_scvi.sh`, run once:

```bash
conda activate worldmodeltraining
pip install scvi-tools
```

Verified locally against `scvi-tools==1.3.3` / `torch==2.13`. If TACC
resolves a different scvi-tools version, re-run `test` mode first
(`bash tacc_commands/TACC_stage1_scvi.sh test`) before a full job — the
`inference()`/`generative()` dict keys this wrapper depends on
(`qz.loc`, `px.scale`) are part of scvi-tools' stable public module API but
have not been checked against every historical version.

## Local validation performed

Both scripts were run end-to-end locally (CPU, `scvi-tools==1.3.3`) against
a synthetic toy dataset built with the same manifest/shard/obs schema as the
real pipeline (sparse `X`/`layers["target"]`, `patient_id`/`dataset_id`
metadata, 5-fold patient-grouped split): Stage-1 `train_stage1_scvi.py`
trained and produced a checkpoint; `ScVIExpressionAutoencoder.from_checkpoint`
reloaded it and gave deterministic, correctly-shaped `encode()`/`decode()`
output; Stage-2 `train_five_level_world_model.py --level 2
--stage1-scvi-checkpoint ...` trained a transition model on top of the frozen
scVI latent without error. This was a correctness smoke test only (tiny
synthetic data, a few steps) — not a real-scale training run.
