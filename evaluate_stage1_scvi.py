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

1. Reconstruction. encode() then decode(), compared against the target in
   whatever units decode() returns: log1p for count likelihoods, and the
   encoder's own input units for the Normal likelihood (log1p_10k, or bin
   indices under a bin_<N> transform).

   This means MSE is only comparable BETWEEN CONFIGURATIONS THAT SHARE UNITS --
   a bin-space MSE and a log1p MSE are different quantities. The per-cell
   Pearson correlation is the cross-configuration metric, since correlation is
   invariant to the scale of either side; it asks whether the model ranks a
   cell's genes correctly, which means the same thing in any units.

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
from scvi_stage1_representation import (
    CHECKPOINT_FORMAT as SCVI_FORMAT,
    ScVIExpressionAutoencoder,
    peek_checkpoint_metadata,
)
from train_stage1_scvi import build_split_anndata


def load_stage1(checkpoint: Path):
    """
    Load either Stage-1 representation and return (model, metadata).

    Both expose the same encode(x, batch_categories) / decode(z, batch_categories)
    contract, so everything downstream is format-agnostic. Dispatch is on the
    checkpoint's own `format` field rather than on a flag, so the two cannot be
    confused.
    """
    payload_format = torch.load(checkpoint, map_location="cpu", weights_only=False).get("format")
    if payload_format == SCVI_FORMAT:
        meta = peek_checkpoint_metadata(checkpoint)
        return ScVIExpressionAutoencoder.from_checkpoint(checkpoint), meta

    from masked_vae import CHECKPOINT_FORMAT as MASKED_FORMAT, MaskedVAERepresentation

    if payload_format == MASKED_FORMAT:
        model = MaskedVAERepresentation.from_checkpoint(checkpoint)
        meta = dict(model.checkpoint_metadata)
        meta.setdefault("gene_likelihood", "masked_gaussian")
        meta["batch_category_to_index"] = model.batch_category_to_index
        return model, meta

    raise ValueError(f"Unrecognised Stage-1 checkpoint format: {payload_format!r}")


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
    p.add_argument(
        "--gene-mask-npz",
        type=Path,
        default=None,
        help="Optional dataset gene-mask npz. When given, reconstruction is scored ONLY "
        "over genes each cell's dataset actually measured. Strongly recommended: a "
        "structural zero is not a real target, so scoring against it measures agreement "
        "with padding. Applies to every configuration equally, so comparability is kept "
        "and a masked-reconstruction model is not penalised on genes it never modelled.",
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

    model, meta = load_stage1(args.checkpoint)
    model = model.to(device)
    model.eval()
    keep_technologies = set(meta["keep_technologies"]) if meta.get("keep_technologies") else None
    print("checkpoint:", args.checkpoint)
    print("  gene_likelihood     :", meta.get("gene_likelihood", "?"))
    print("  expression_transform:", meta.get("expression_transform", "none"))
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
        expression_transform=meta.get("expression_transform", "none"),
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

    batch_categories_all = (
        val_adata.obs[meta["batch_key"]].astype(str).tolist() if meta["batch_key"] else None
    )

    # The reconstruction target must be in whatever units decode() returns,
    # which depends on the likelihood:
    #   count likelihoods -> log1p(px_scale * reference_library), so the target
    #                        is log1p even though the encoder consumed counts
    #   normal likelihood -> px.loc, i.e. the same units the encoder consumed
    #                        (log1p_10k, or bin indices for a bin_<N> transform)
    # Getting this wrong silently scores predictions against the wrong scale --
    # comparing bin-space output to a log1p target, for instance.
    from sample_paired_h5ad_dataloader import transform_expression_sparse

    if meta.get("expression_transform", "none") == "none":
        target_matrix = transform_expression_sparse(val_adata.X, "log1p_10k")
        recon_units = "log1p_10k"
    else:
        target_matrix = val_adata.X
        recon_units = meta.get("expression_transform", "none")
    print("  reconstruction units:", recon_units)

    # Optional per-cell measured-gene mask. Scoring reconstruction against a
    # structural zero measures agreement with padding, not with biology, so
    # when a mask is available every configuration is scored only where there
    # is a real observation.
    gene_mask_rows = None
    if args.gene_mask_npz is not None:
        from train_stage1_masked_vae import build_mask_matrix, load_gene_masks

        mask_index, mask_matrix = load_gene_masks(args.gene_mask_npz, list(meta["gene_ids"]))
        gene_mask_rows = build_mask_matrix(
            val_adata.obs["dataset_id"].to_numpy(), mask_index, mask_matrix
        )
        print(
            f"  scoring reconstruction over measured genes only "
            f"(mean {gene_mask_rows.mean():.3f} of the panel per cell)"
        )

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
            if gene_mask_rows is None:
                mask = torch.ones_like(target)
            else:
                mask = torch.as_tensor(
                    gene_mask_rows[start:end].astype(np.float32), device=device
                )

            squared_error += float((((y - target) ** 2) * mask).sum())
            n_values += float(mask.sum())

            # Center and correlate over the masked entries only, so unmeasured
            # genes influence neither the means nor the correlation.
            counts = mask.sum(dim=1, keepdim=True).clamp(min=1.0)
            yc = (y - (y * mask).sum(dim=1, keepdim=True) / counts) * mask
            tc = (target - (target * mask).sum(dim=1, keepdim=True) / counts) * mask
            denom = yc.norm(dim=1) * tc.norm(dim=1)
            r = torch.where(denom > 0, (yc * tc).sum(dim=1) / denom.clamp(min=1e-8),
                            torch.zeros_like(denom))
            cell_pearson.append(r.cpu().numpy())

    latent = np.concatenate(latents, axis=0)
    pearson = np.concatenate(cell_pearson, axis=0)
    recon_mse = squared_error / max(n_values, 1)

    print()
    print("=" * 64)
    print(f"RECONSTRUCTION (units: {recon_units})")
    print("=" * 64)
    print(f"  MSE                      : {recon_mse:.6f}   <- only comparable within the same units")
    print(f"  per-cell Pearson r  mean : {float(pearson.mean()):.4f}   <- comparable across all configs")
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
        "gene_likelihood": meta.get("gene_likelihood", "?"),
        "expression_transform": meta.get("expression_transform", "none"),
        "batch_key": meta["batch_key"],
        "n_cells": int(val_adata.n_obs),
        "recon_units": recon_units,
        "recon_mse": recon_mse,
        "recon_mse_log1p": recon_mse if recon_units == "log1p_10k" else None,
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
