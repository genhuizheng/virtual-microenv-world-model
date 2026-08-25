#!/usr/bin/env python3
"""Train Stage 1 as a real scVI negative-binomial VAE on raw-count H5AD pairs.

Replaces the ad hoc GaussianExpressionVAE Stage 1 (train_stage1_gaussian_vae.py)
with an actual scvi-tools model. Source and target cells from every shard are
pooled into one in-memory AnnData (raw counts, no expression transform) and
scVI is trained with its native NB/ZINB reconstruction likelihood.

The patient-level train/val split uses compute_group_to_fold() from
sample_paired_h5ad_dataloader.py, so this Stage-1 fold assignment is
byte-identical to the fold assignment train_five_level_world_model.py will
later use for Stage 2 with the same --group-column/--num-folds/--seed -- no
patient ever leaks between what scVI was trained on and what a transition
model is validated on.

Only the trained torch module (scvi.module.VAE) state_dict plus its plain
constructor kwargs are saved (format "stage1_scvi_v1"), not a
scvi.model.SCVI bundle -- this avoids carrying a full AnnData/registry into
Stage 2, which may run in a different process or node. See
scvi_stage1_representation.py for the encode()/decode() contract this
checkpoint feeds into.

Requires --expression-transform none in Stage 2 (raw counts): scVI applies
its own internal log1p transform before the encoder, and its reconstruction
likelihood is defined on counts, not on already log1p_10k-transformed data.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import anndata as ad
import torch

from sample_paired_h5ad_dataloader import compute_group_to_fold, safe_get_gene_symbols
from scvi_stage1_representation import CHECKPOINT_FORMAT


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--checkpoint-dir", type=Path, required=True)
    p.add_argument("--max-epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--latent-dim", type=int, default=128)
    p.add_argument("--n-hidden", type=int, default=128)
    p.add_argument("--n-layers", type=int, default=1)
    p.add_argument("--dropout-rate", type=float, default=0.1)
    p.add_argument(
        "--dispersion",
        choices=["gene", "gene-batch", "gene-label", "gene-cell"],
        default="gene",
    )
    p.add_argument("--gene-likelihood", choices=["zinb", "nb", "poisson"], default="zinb")
    p.add_argument("--reference-library-size", type=float, default=1e4)
    p.add_argument("--num-folds", type=int, default=5)
    p.add_argument("--fold-index", type=int, default=0)
    p.add_argument("--group-column", type=str, default="patient_id")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--max-cells",
        type=int,
        default=-1,
        help="Cap total pooled source+target cells per split; -1 uses everything. "
        "For quick smoke tests only.",
    )
    p.add_argument("--early-stopping", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def load_paired_shards_as_role_anndata(data_dir: Path) -> ad.AnnData:
    """Pool every shard's source (adata.X) and target (adata.layers['target'])
    rows into one AnnData with an obs['role'] column, preserving sparsity."""
    manifest_path = data_dir / "paired_h5ad_manifest.tsv"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing manifest: {manifest_path}")
    manifest = pd.read_csv(manifest_path, sep="\t")
    if manifest.empty:
        raise ValueError(f"Empty manifest: {manifest_path}")

    source_blocks = []
    target_blocks = []
    obs_rows = []
    gene_ids = None
    var_table = None

    for rel in manifest["relative_shard_file"].astype(str):
        shard_path = data_dir / rel
        if not shard_path.exists():
            raise FileNotFoundError(f"Missing shard: {shard_path}")
        shard = ad.read_h5ad(shard_path)
        if "target" not in shard.layers:
            raise ValueError(f"Missing target layer in shard: {shard_path}")

        this_gene_ids = list(shard.var_names.astype(str))
        if gene_ids is None:
            gene_ids = this_gene_ids
            var_table = shard.var.copy()
        elif this_gene_ids != gene_ids:
            raise ValueError(f"Gene order mismatch in shard: {shard_path}")

        x = shard.X
        y = shard.layers["target"]
        source_blocks.append(x if sp.issparse(x) else sp.csr_matrix(x))
        target_blocks.append(y if sp.issparse(y) else sp.csr_matrix(y))
        obs_rows.append(shard.obs.copy())

    obs = pd.concat(obs_rows, axis=0, ignore_index=True)
    combined_x = sp.vstack(
        [sp.vstack(source_blocks, format="csr"), sp.vstack(target_blocks, format="csr")],
        format="csr",
    )
    combined_obs = pd.concat(
        [obs.assign(role="source"), obs.assign(role="target")],
        axis=0,
        ignore_index=True,
    )

    combined = ad.AnnData(X=combined_x, obs=combined_obs, var=var_table.copy())
    combined.var_names = gene_ids
    return combined


def build_split_anndata(
    data_dir: Path,
    group_column: str,
    num_folds: int,
    fold_index: int,
    seed: int,
) -> tuple[ad.AnnData, ad.AnnData]:
    combined = load_paired_shards_as_role_anndata(data_dir)
    if num_folds <= 1:
        return combined, combined[:0].copy()

    group_to_fold = compute_group_to_fold(data_dir, group_column, num_folds, seed)
    if group_column not in combined.obs.columns:
        raise ValueError(f"Missing group column {group_column!r} in shard obs")
    folds = combined.obs[group_column].astype(str).map(group_to_fold)
    if folds.isna().any():
        raise ValueError(f"Unassigned groups for column {group_column!r}")
    combined.obs["_fold"] = folds.astype(int)
    train_adata = combined[combined.obs["_fold"] != fold_index].copy()
    val_adata = combined[combined.obs["_fold"] == fold_index].copy()
    if train_adata.n_obs == 0 or val_adata.n_obs == 0:
        raise ValueError(f"Empty split for num_folds={num_folds}, fold_index={fold_index}")
    return train_adata, val_adata


def subsample(adata: ad.AnnData, max_cells: int, seed: int) -> ad.AnnData:
    if max_cells <= 0 or adata.n_obs <= max_cells:
        return adata
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(adata.n_obs, size=max_cells, replace=False))
    return adata[idx].copy()


def main() -> None:
    args = parse_args()
    if not 0 <= args.fold_index < args.num_folds:
        raise ValueError("--fold-index must be in [0, --num-folds)")
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_path = args.checkpoint_dir / f"best_stage1_scvi_fold{args.fold_index}.pt"
    history_path = args.checkpoint_dir / f"stage1_scvi_fold{args.fold_index}_history.csv"
    config_path = args.checkpoint_dir / f"stage1_scvi_fold{args.fold_index}_config.json"
    if not args.overwrite and (best_path.exists() or history_path.exists()):
        raise FileExistsError("Stage-1 scVI outputs already exist; use --overwrite")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    import scvi  # heavy dependency; only imported once this script actually runs

    print("scvi version:", scvi.__version__)
    print("torch version:", torch.__version__)

    train_adata, val_adata = build_split_anndata(
        args.data_dir, args.group_column, args.num_folds, args.fold_index, args.seed
    )
    train_adata = subsample(train_adata, args.max_cells, args.seed)
    val_adata = subsample(val_adata, args.max_cells, args.seed + 1)

    print(
        f"train_cells={train_adata.n_obs} val_cells={val_adata.n_obs} "
        f"n_genes={train_adata.n_vars} fold={args.fold_index}/{args.num_folds}"
    )

    gene_ids = list(train_adata.var_names.astype(str))
    gene_symbols = safe_get_gene_symbols(train_adata.var, gene_ids)

    scvi.model.SCVI.setup_anndata(train_adata)
    use_gpu = args.device in ("auto", "cuda") and torch.cuda.is_available()
    accelerator = "gpu" if use_gpu else "cpu"

    model = scvi.model.SCVI(
        train_adata,
        n_latent=args.latent_dim,
        n_hidden=args.n_hidden,
        n_layers=args.n_layers,
        dropout_rate=args.dropout_rate,
        dispersion=args.dispersion,
        gene_likelihood=args.gene_likelihood,
    )
    model.train(
        max_epochs=args.max_epochs,
        batch_size=args.batch_size,
        early_stopping=args.early_stopping,
        accelerator=accelerator,
        devices=1,
        plan_kwargs={"lr": args.lr},
    )

    val_reconstruction_error = None
    if val_adata.n_obs > 0:
        val_metrics = model.get_reconstruction_error(adata=val_adata)
        val_reconstruction_error = float(val_metrics["reconstruction_loss"])
        print(f"val_reconstruction_loss={val_reconstruction_error:.6g}")
    else:
        print("num_folds=1: no held-out validation split (full-data training).")

    n_batch = int(model.summary_stats["n_batch"])
    module_init_kwargs = dict(
        n_input=train_adata.n_vars,
        n_batch=n_batch,
        n_hidden=args.n_hidden,
        n_latent=args.latent_dim,
        n_layers=args.n_layers,
        dropout_rate=args.dropout_rate,
        dispersion=args.dispersion,
        gene_likelihood=args.gene_likelihood,
        latent_distribution="normal",
        log_variational=True,
        use_observed_lib_size=True,
        use_batch_norm="both",
        use_layer_norm="none",
    )

    checkpoint = {
        "format": CHECKPOINT_FORMAT,
        "module_state_dict": model.module.state_dict(),
        "module_init_kwargs": module_init_kwargs,
        "reference_library_size": args.reference_library_size,
        "n_genes": train_adata.n_vars,
        "gene_ids": gene_ids,
        "gene_symbols": gene_symbols,
        "fold_index": args.fold_index,
        "num_folds": args.num_folds,
        "group_column": args.group_column,
        "val_reconstruction_error": val_reconstruction_error,
        "scvi_version": scvi.__version__,
        "torch_version": torch.__version__,
        "args": vars(args),
    }
    torch.save(checkpoint, best_path)

    history_row = {
        "epoch": args.max_epochs,
        "train_cells": train_adata.n_obs,
        "val_cells": val_adata.n_obs,
        "val_reconstruction_loss": val_reconstruction_error,
    }
    with history_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history_row))
        writer.writeheader()
        writer.writerow(history_row)
    config_path.write_text(json.dumps(vars(args), indent=2, default=str) + "\n", encoding="utf-8")

    print(f"best_stage1_scvi_checkpoint={best_path}")
    if val_reconstruction_error is not None:
        print(f"val_reconstruction_error={val_reconstruction_error:.6g}")


if __name__ == "__main__":
    main()
