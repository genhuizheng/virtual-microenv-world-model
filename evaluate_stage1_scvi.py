#!/usr/bin/env python3
"""Evaluate a Stage-1 scVI checkpoint on held-out patients.

Why this exists
---------------
Two Stage-1 configurations are under comparison:

    A  raw counts   + zinb/nb/poisson   (standard scVI)
    B  log1p_10k    + normal            (the Chreode configuration)

Their training objectives are not comparable: A reports a negative-binomial
log-likelihood on counts, B a Gaussian one on log1p values. A smaller number
from B means nothing. This script scores both in units they share, so the
comparison is actually meaningful.

Three questions, which is what a batch-conditioned VAE has to get right:

1. Reconstruction. encode() then decode() and compare against the true log1p
   expression. `ScVIExpressionAutoencoder.decode()` returns log1p-scale output
   for every likelihood by construction, so this metric is identical in
   meaning across configs.

2. Batch mixing. For each cell, what fraction of its k nearest latent
   neighbours come from the same source study? If the batch key worked, this
   sits near the chance rate. Well above chance means the latent is still
   organised by study -- batch effects survived.

3. Biological signal. The same kNN statistic for pre/post treatment role.
   This one should stay WELL ABOVE chance: it is the signal the world model
   exists to learn. A configuration that mixes batches by destroying the
   treatment signal has not solved the problem, it has erased the task.

(2) and (3) together are the real test -- either alone is easy to win by
degrading the other.

Usage
-----
    python evaluate_stage1_scvi.py \\
        --checkpoint stage1_A_counts_zinb/best_stage1_scvi_fold0.pt \\
        --data-dir paired_training_h5ad_50k \\
        --technology-lookup ../Virtual_microenv/dataset_platform_lookup.tsv \\
        --num-folds 5 --fold-index 0 --max-cells 20000 --device cuda
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from sample_paired_h5ad_dataloader import derive_source_study
from scvi_stage1_representation import ScVIExpressionAutoencoder, peek_checkpoint_metadata
from train_stage1_scvi import build_split_anndata


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--technology-lookup", type=Path, default=None)
    p.add_argument("--num-folds", type=int, default=5)
    p.add_argument("--fold-index", type=int, default=0)
    p.add_argument("--group-column", type=str, default="patient_id")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--max-cells",
        type=int,
        default=20000,
        help="Subsample the held-out split before the O(n^2)-ish kNN step.",
    )
    p.add_argument("--n-neighbors", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--out", type=Path, default=None, help="Optional JSON output path.")
    return p.parse_args()


def choose_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def knn_same_label_fraction(
    latent: np.ndarray, labels: np.ndarray, n_neighbors: int
) -> tuple[float, float]:
    """
    Mean fraction of k nearest neighbours sharing a cell's label, and the
    chance rate expected if the latent carried no information about it.

    The chance rate is sum(p_i^2) over label proportions -- the probability
    that two independently drawn cells share a label. Comparing against it
    makes the statistic interpretable regardless of how skewed the labels are.
    """
    from sklearn.neighbors import NearestNeighbors

    codes, counts = np.unique(labels, return_counts=True)
    proportions = counts / counts.sum()
    chance = float((proportions**2).sum())

    k = min(n_neighbors, len(latent) - 1)
    if k < 1:
        return float("nan"), chance
    nn = NearestNeighbors(n_neighbors=k + 1).fit(latent)
    _, indices = nn.kneighbors(latent)
    neighbor_labels = labels[indices[:, 1:]]  # drop self
    same = (neighbor_labels == labels[:, None]).mean()
    return float(same), chance


def main() -> None:
    args = parse_args()
    device = choose_device(args.device)

    meta = peek_checkpoint_metadata(args.checkpoint)
    keep_technologies = set(meta["keep_technologies"]) if meta["keep_technologies"] else None
    print("checkpoint:", args.checkpoint)
    print("  gene_likelihood     :", meta["gene_likelihood"])
    print("  expression_transform:", meta["expression_transform"])
    print("  batch_key           :", meta["batch_key"])
    print("  keep_technologies   :", meta["keep_technologies"])

    # Rebuild the identical split Stage 1 used, so "held out" means the same
    # patients here as it did during training. The checkpoint's gene_ids are the
    # genes that survived Stage 1's panel restriction AND its detection filter,
    # so passing them as the panel reproduces the exact input width and order
    # the encoder was fitted on; min_cells_detected is therefore left at 0.
    print("  genes               :", meta["n_genes"])
    _, val_adata = build_split_anndata(
        args.data_dir,
        args.group_column,
        args.num_folds,
        args.fold_index,
        args.seed,
        technology_lookup_path=args.technology_lookup,
        keep_technologies=keep_technologies,
        expression_transform=meta["expression_transform"],
        keep_gene_ids=list(meta["gene_ids"]),
    )
    if val_adata.n_vars != meta["n_genes"]:
        raise ValueError(
            f"rebuilt {val_adata.n_vars} genes but the checkpoint expects {meta['n_genes']}"
        )
    if val_adata.n_obs == 0:
        raise ValueError("Validation split is empty")

    known = meta["batch_category_to_index"]
    if known and meta["batch_key"]:
        keep = val_adata.obs[meta["batch_key"]].astype(str).isin(set(known)).to_numpy()
        dropped = int((~keep).sum())
        if dropped:
            print(
                f"  dropping {dropped}/{val_adata.n_obs} held-out cells whose "
                f"{meta['batch_key']} was never trained on"
            )
        val_adata = val_adata[keep].copy()
        if val_adata.n_obs == 0:
            raise ValueError("Every held-out cell has an untrained batch category")

    if 0 < args.max_cells < val_adata.n_obs:
        rng = np.random.default_rng(args.seed)
        idx = np.sort(rng.choice(val_adata.n_obs, size=args.max_cells, replace=False))
        val_adata = val_adata[idx].copy()
    print(f"  evaluating on {val_adata.n_obs} held-out cells")

    model = ScVIExpressionAutoencoder.from_checkpoint(args.checkpoint).to(device)
    model.eval()

    batch_categories_all = (
        val_adata.obs[meta["batch_key"]].astype(str).tolist() if meta["batch_key"] else None
    )

    # The reconstruction target is always log1p-scale, matching decode()'s
    # contract, regardless of which units the encoder consumes.
    from sample_paired_h5ad_dataloader import transform_expression_sparse

    if meta["expression_transform"] == "log1p_10k":
        target_matrix = val_adata.X  # already log1p
    else:
        target_matrix = transform_expression_sparse(val_adata.X, "log1p_10k")

    latents, cell_pearson, squared_error, n_values = [], [], 0.0, 0
    with torch.no_grad():
        for start in range(0, val_adata.n_obs, args.batch_size):
            end = min(start + args.batch_size, val_adata.n_obs)
            x_in = torch.as_tensor(
                np.asarray(val_adata.X[start:end].todense(), dtype=np.float32), device=device
            )
            cats = batch_categories_all[start:end] if batch_categories_all else None
            z = model.encode(x_in, cats)
            y = model.decode(z, cats)
            latents.append(z.cpu().numpy())

            target = torch.as_tensor(
                np.asarray(target_matrix[start:end].todense(), dtype=np.float32), device=device
            )
            squared_error += float(((y - target) ** 2).sum())
            n_values += target.numel()

            yc = y - y.mean(dim=1, keepdim=True)
            tc = target - target.mean(dim=1, keepdim=True)
            denom = yc.norm(dim=1) * tc.norm(dim=1)
            r = torch.where(denom > 0, (yc * tc).sum(dim=1) / denom.clamp(min=1e-8),
                            torch.zeros_like(denom))
            cell_pearson.append(r.cpu().numpy())

    latent = np.concatenate(latents, axis=0)
    pearson = np.concatenate(cell_pearson, axis=0)
    recon_mse = squared_error / max(n_values, 1)

    print()
    print("=" * 64)
    print("RECONSTRUCTION (log1p units -- comparable across configurations)")
    print("=" * 64)
    print(f"  MSE                      : {recon_mse:.6f}")
    print(f"  per-cell Pearson r  mean : {float(pearson.mean()):.4f}")
    print(f"                    median : {float(np.median(pearson)):.4f}")

    source_study = np.array(
        [derive_source_study(p) for p in val_adata.obs["patient_id"].astype(str)]
    )
    study_same, study_chance = knn_same_label_fraction(latent, source_study, args.n_neighbors)

    print()
    print("=" * 64)
    print(f"BATCH MIXING (k={args.n_neighbors} nearest latent neighbours)")
    print("=" * 64)
    print(f"  same source_study fraction: {study_same:.4f}")
    print(f"  chance rate               : {study_chance:.4f}")
    print(f"  ratio (1.0 = fully mixed) : {study_same / max(study_chance, 1e-8):.2f}x")
    print("  lower is better; >> 1 means the latent is still organised by study")

    role_same = role_chance = float("nan")
    if "role" in val_adata.obs.columns:
        role = val_adata.obs["role"].astype(str).to_numpy()
        role_same, role_chance = knn_same_label_fraction(latent, role, args.n_neighbors)
        print()
        print("=" * 64)
        print("BIOLOGICAL SIGNAL (pre/post treatment role)")
        print("=" * 64)
        print(f"  same role fraction        : {role_same:.4f}")
        print(f"  chance rate               : {role_chance:.4f}")
        print(f"  ratio (1.0 = erased)      : {role_same / max(role_chance, 1e-8):.2f}x")
        print("  HIGHER is better here; near 1.0 means treatment signal was destroyed")

    result = {
        "checkpoint": str(args.checkpoint),
        "gene_likelihood": meta["gene_likelihood"],
        "expression_transform": meta["expression_transform"],
        "batch_key": meta["batch_key"],
        "n_cells": int(val_adata.n_obs),
        "recon_mse_log1p": recon_mse,
        "recon_pearson_mean": float(pearson.mean()),
        "recon_pearson_median": float(np.median(pearson)),
        "knn_same_source_study": study_same,
        "knn_source_study_chance": study_chance,
        "knn_same_role": role_same,
        "knn_role_chance": role_chance,
    }
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
