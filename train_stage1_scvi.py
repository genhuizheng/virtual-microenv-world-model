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

from sample_paired_h5ad_dataloader import (
    compute_group_to_fold,
    derive_source_study,
    load_technology_lookup,
    normalize_technology,
    parse_bin_transform,
    safe_get_gene_symbols,
    transform_expression_sparse,
)
from scvi_stage1_representation import CHECKPOINT_FORMAT, resolve_decode_units


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
    p.add_argument(
        "--gene-likelihood",
        choices=["zinb", "nb", "poisson", "normal"],
        default="zinb",
        help="Count likelihoods (zinb/nb/poisson) require raw counts -- the standard "
        "scVI setup, and the default here. 'normal' requires "
        "--expression-transform log1p_10k instead: it is the configuration the "
        "Chreode paper uses (normalize_total(1e4)+log1p with a Normal likelihood). "
        "Mixing the two -- log1p input with a count likelihood, or raw counts with "
        "'normal' -- is silently wrong and is rejected.",
    )
    p.add_argument(
        "--expression-transform",
        type=str,
        default="none",
        help="One of: 'none' (raw counts; required for count likelihoods -- scVI applies "
        "its own internal log1p before the encoder and models library size explicitly), "
        "'log1p_10k' (normalize_total(1e4)+log1p), or 'bin_<N>' e.g. 'bin_51' "
        "(scGPT-style per-cell rank binning into N equal-count bins). Both log1p_10k and "
        "bin_<N> require --gene-likelihood normal, since neither produces counts.",
    )
    p.add_argument("--reference-library-size", type=float, default=1e4)
    p.add_argument("--num-folds", type=int, default=5)
    p.add_argument("--fold-index", type=int, default=0)
    p.add_argument("--group-column", type=str, default="patient_id")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--batch-key",
        choices=["none", "source_study", "technology", "dataset_id", "patient_id"],
        default="none",
        help="Technical batch covariate for scVI. 'source_study' (the GEO/ArrayExpress "
        "accession embedded in patient_id) is recommended: one study means one lab, one "
        "protocol, one technology. Do NOT use 'dataset_id' -- scCT-DB datasets are "
        "partitioned by therapeutic regimen, cancer subtype, and drug-response group, so "
        "conditioning on it would remove the biology this project models.",
    )
    p.add_argument(
        "--technology-lookup",
        type=Path,
        default=None,
        help="TSV with dataset_id/technology columns. Required for --keep-technologies "
        "and for --batch-key technology.",
    )
    p.add_argument(
        "--keep-technologies",
        type=str,
        default=None,
        help="Comma-separated technologies to retain (case-insensitive), e.g. "
        "'10x genomics'. Rows from other technologies are dropped before training.",
    )
    p.add_argument(
        "--keep-gene-list",
        type=Path,
        default=None,
        help="File of gene ids to restrict the panel to, one per line (or a TSV with a "
        "gene_id column). Applied before --min-cells-detected and before the expression "
        "transform. Use this to replace the zero-filled union panel with genes that most "
        "datasets actually measured -- structural zeros are a per-study fingerprint the "
        "model would otherwise learn as biology.",
    )
    p.add_argument(
        "--min-cells-detected",
        type=int,
        default=0,
        help="Drop genes detected (nonzero) in fewer than this many TRAINING cells. "
        "0 disables filtering. Use 1 to remove genes never observed in any cell: the "
        "unified panel is a zero-filled union across studies, so a large share of its "
        "columns are structural zeros. They carry no information, inflate the input "
        "projection (which dominates the parameter count at 46k genes), and let a "
        "Gaussian likelihood collapse its variance to zero on them. Computed on the "
        "training split only, so it leaks nothing from held-out patients.",
    )
    p.add_argument(
        "--max-cells",
        type=int,
        default=-1,
        help="Cap total pooled source+target cells per split; -1 uses everything. "
        "For quick smoke tests only.",
    )
    p.add_argument(
        "--decode-units",
        choices=["auto", "counts", "log1p_10k"],
        default="auto",
        help="Units decode() returns. 'auto' mirrors --expression-transform, giving a "
        "symmetric autoencoder: raw counts in -> raw counts out, log1p_10k in -> "
        "log1p_10k out. Set explicitly to decouple them (e.g. counts in, log1p out, "
        "which is what checkpoints written before this flag existed always did).",
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


def annotate_batch_metadata(
    combined: ad.AnnData,
    technology_lookup_path: Path | None,
    keep_technologies: set[str] | None,
) -> ad.AnnData:
    """
    Add `source_study` (and `technology`, when a lookup is given) to obs, then
    optionally drop rows from unwanted technologies.

    `source_study` is the GEO/ArrayExpress accession embedded in patient_id:
    one study means one lab, one protocol, one sequencing technology, which is
    the correct *technical* batch unit. dataset_id is deliberately not used for
    this -- scCT-DB partitions datasets by therapeutic regimen, cancer subtype,
    and drug-response group, all of which are biological.
    """
    if "patient_id" not in combined.obs.columns:
        raise ValueError("pooled obs has no patient_id column; cannot derive source_study")
    combined.obs["source_study"] = [
        derive_source_study(p) for p in combined.obs["patient_id"].astype(str)
    ]

    if technology_lookup_path is not None:
        lookup = load_technology_lookup(technology_lookup_path)
        if "dataset_id" not in combined.obs.columns:
            raise ValueError("pooled obs has no dataset_id column; cannot resolve technology")
        technologies = (
            combined.obs["dataset_id"].astype(str).map(lambda d: lookup.get(d.strip(), ""))
        )
        unknown = technologies == ""
        if unknown.any():
            missing = sorted(set(combined.obs.loc[unknown, "dataset_id"].astype(str)))[:5]
            raise ValueError(
                f"{int(unknown.sum())} cell(s) have a dataset_id absent from the technology "
                f"lookup, e.g. {missing}. Refusing to proceed on incomplete metadata."
            )
        combined.obs["technology"] = technologies.to_numpy()

        if keep_technologies is not None:
            keep_mask = combined.obs["technology"].isin(keep_technologies).to_numpy()
            dropped = int((~keep_mask).sum())
            kept_counts = combined.obs.loc[~keep_mask, "technology"].value_counts()
            print(
                f"technology filter: keeping {sorted(keep_technologies)} -> "
                f"{int(keep_mask.sum())} cells kept, {dropped} dropped"
            )
            if dropped:
                print("  dropped by technology:")
                print("   ", kept_counts.to_dict())
            if not keep_mask.any():
                raise ValueError("technology filter removed every cell")
            combined = combined[keep_mask].copy()
    elif keep_technologies is not None:
        raise ValueError("--keep-technologies requires --technology-lookup")

    return combined


def build_split_anndata(
    data_dir: Path,
    group_column: str,
    num_folds: int,
    fold_index: int,
    seed: int,
    technology_lookup_path: Path | None = None,
    keep_technologies: set[str] | None = None,
    expression_transform: str = "none",
    min_cells_detected: int = 0,
    keep_gene_ids: list[str] | None = None,
) -> tuple[ad.AnnData, ad.AnnData]:
    combined = load_paired_shards_as_role_anndata(data_dir)
    combined = annotate_batch_metadata(combined, technology_lookup_path, keep_technologies)

    if num_folds <= 1:
        train_adata, val_adata = combined, combined[:0].copy()
    else:
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

    # Panel restriction. This is a fixed, externally supplied list (e.g. genes
    # measured by >=90% of source datasets), so unlike the detection filter it
    # uses no information from the data at all and cannot leak.
    if keep_gene_ids is not None:
        wanted = [str(g) for g in keep_gene_ids]
        available = set(train_adata.var_names.astype(str))
        missing = [g for g in wanted if g not in available]
        if missing:
            raise ValueError(
                f"{len(missing)} gene(s) in --keep-gene-list are absent from the data, "
                f"e.g. {missing[:5]}"
            )
        print(f"panel restriction: {len(wanted)}/{train_adata.n_vars} genes kept")
        train_adata = train_adata[:, wanted].copy()
        if val_adata.n_obs:
            val_adata = val_adata[:, wanted].copy()
        else:
            val_adata = val_adata[:, wanted].copy()

    # Gene filtering, computed on the TRAINING split only so that held-out
    # patients contribute nothing to the choice of genes, and applied BEFORE
    # the expression transform so normalize_total's library is the sum over the
    # genes actually modelled. The Stage-2 dataloader subsets in the same order.
    if min_cells_detected > 0:
        train_counts = np.asarray((train_adata.X > 0).sum(axis=0)).ravel()
        keep = train_counts >= min_cells_detected
        n_dropped = int((~keep).sum())
        never = int((train_counts == 0).sum())
        print(
            f"gene filter (>= {min_cells_detected} training cells): "
            f"{int(keep.sum())}/{len(keep)} kept, {n_dropped} dropped "
            f"({never} never detected)"
        )
        if not keep.any():
            raise ValueError("Gene filter removed every gene")
        train_adata = train_adata[:, keep].copy()
        val_adata = val_adata[:, keep].copy() if val_adata.n_obs else val_adata[:, keep].copy()

    if expression_transform != "none":
        print(f"applying expression transform {expression_transform!r}")
    # Same function family the Stage-2 dataloader uses, so both stages feed the
    # frozen encoder identical units.
    train_adata.X = transform_expression_sparse(train_adata.X, expression_transform)
    if val_adata.n_obs:
        val_adata.X = transform_expression_sparse(val_adata.X, expression_transform)
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

    # The likelihood and the input transform must agree. A count likelihood is
    # defined on non-negative integers; a Normal likelihood is not. scvi-tools
    # does not validate this, so a mismatch would train "successfully" while
    # modelling the wrong thing.
    count_likelihoods = {"zinb", "nb", "poisson"}
    is_binned = parse_bin_transform(args.expression_transform) is not None
    if args.expression_transform not in {"none", "log1p_10k"} and not is_binned:
        raise ValueError(
            f"Unknown --expression-transform {args.expression_transform!r}; expected "
            "'none', 'log1p_10k', or 'bin_<N>'."
        )
    if args.gene_likelihood in count_likelihoods and args.expression_transform != "none":
        raise ValueError(
            f"--gene-likelihood {args.gene_likelihood} is a count distribution and requires "
            "--expression-transform none (raw counts). scVI applies its own log1p before "
            "the encoder and models library size explicitly."
        )
    if args.gene_likelihood == "normal" and args.expression_transform == "none":
        raise ValueError(
            "--gene-likelihood normal requires --expression-transform log1p_10k or "
            "bin_<N>. A Normal likelihood on raw counts models the wrong scale."
        )
    # Resolved once here so the printed value, the checkpoint field and the
    # wrapper's runtime behaviour cannot disagree.
    decode_units = resolve_decode_units(args.decode_units, args.expression_transform)
    print(
        f"units: input={args.expression_transform!r} -> decode()={decode_units!r}"
        + ("  (symmetric)" if args.decode_units == "auto" else "  (explicit)")
    )
    if is_binned:
        # Bin indices are ranks, not abundances: they have no library size, and
        # scVI's decoder still applies library scaling. mean_log_library (used
        # for the Normal decode path) therefore ends up standing in for "typical
        # total bin mass per cell". It is well-defined and the evaluator scores
        # it fairly, but if binning wins decisively it deserves a proper
        # implementation outside scVI rather than this accommodation.
        print(
            f"NOTE: {args.expression_transform} feeds rank bins to scVI, whose decoder "
            "assumes library-scaled abundances. See train_stage1_scvi.py for why."
        )

    keep_technologies = None
    if args.keep_technologies:
        keep_technologies = {
            normalize_technology(t) for t in args.keep_technologies.split(",") if t.strip()
        }
        if not keep_technologies:
            raise ValueError("--keep-technologies was given but parsed to an empty set")

    batch_key = None if args.batch_key == "none" else args.batch_key
    if batch_key == "technology" and args.technology_lookup is None:
        raise ValueError("--batch-key technology requires --technology-lookup")
    if batch_key == "dataset_id":
        print(
            "WARNING: --batch-key dataset_id conditions on a variable partitioned by "
            "therapeutic regimen, cancer subtype, and drug-response group. This removes "
            "biological signal this project models. Prefer --batch-key source_study."
        )

    keep_gene_ids = None
    if args.keep_gene_list is not None:
        text = args.keep_gene_list.read_text(encoding="utf-8").splitlines()
        rows = [line.strip() for line in text if line.strip()]
        # tolerate a TSV with a gene_id header column
        if rows and rows[0].lower().split("	")[0] in {"gene_id", "gene"}:
            rows = rows[1:]
        keep_gene_ids = [r.split("	")[0] for r in rows]
        if not keep_gene_ids:
            raise ValueError(f"--keep-gene-list {args.keep_gene_list} is empty")
        if len(set(keep_gene_ids)) != len(keep_gene_ids):
            raise ValueError("--keep-gene-list contains duplicate gene ids")
        print(f"keep_gene_list: {len(keep_gene_ids)} genes from {args.keep_gene_list}")

    train_adata, val_adata = build_split_anndata(
        args.data_dir,
        args.group_column,
        args.num_folds,
        args.fold_index,
        args.seed,
        technology_lookup_path=args.technology_lookup,
        keep_technologies=keep_technologies,
        expression_transform=args.expression_transform,
        min_cells_detected=args.min_cells_detected,
        keep_gene_ids=keep_gene_ids,
    )
    train_adata = subsample(train_adata, args.max_cells, args.seed)
    val_adata = subsample(val_adata, args.max_cells, args.seed + 1)

    print(
        f"train_cells={train_adata.n_obs} val_cells={val_adata.n_obs} "
        f"n_genes={train_adata.n_vars} fold={args.fold_index}/{args.num_folds}"
    )

    gene_ids = list(train_adata.var_names.astype(str))
    gene_symbols = safe_get_gene_symbols(train_adata.var, gene_ids)

    if batch_key is not None:
        n_categories = train_adata.obs[batch_key].nunique()
        print(f"batch_key={batch_key}  ({n_categories} categories in train split)")
        if n_categories < 2:
            raise ValueError(
                f"--batch-key {batch_key} has only {n_categories} category after "
                "filtering; a single-category batch key corrects nothing. Either drop "
                "--batch-key or choose a finer grouping (e.g. source_study)."
            )
        # A batch category present only in validation has no trained embedding,
        # so scVI cannot embed those cells at all -- get_reconstruction_error()
        # raises rather than warns. This is a real consequence of combining a
        # study-level batch key with patient-level folds: a study whose patients
        # all land in the held-out fold is never trained on. Drop those cells
        # and say so, rather than reporting a number computed on random
        # embeddings.
        train_categories = set(train_adata.obs[batch_key].astype(str))
        val_categories = set(val_adata.obs[batch_key].astype(str))
        unseen = val_categories - train_categories
        if unseen and val_adata.n_obs:
            keep = val_adata.obs[batch_key].astype(str).isin(train_categories).to_numpy()
            dropped = int((~keep).sum())
            print(
                f"WARNING: {len(unseen)} {batch_key} value(s) occur only in the "
                f"validation split and were never trained on: {sorted(unseen)[:8]}"
                f"{' ...' if len(unseen) > 8 else ''}"
            )
            print(
                f"  dropping {dropped}/{val_adata.n_obs} validation cells "
                f"({100.0 * dropped / val_adata.n_obs:.2f}%) that belong to them; "
                "validation error is reported on the remainder."
            )
            val_adata = val_adata[keep].copy()
            if val_adata.n_obs == 0:
                raise ValueError(
                    f"Every validation cell belongs to a {batch_key} unseen in training. "
                    "Use a coarser --batch-key, or a fold assignment that keeps each "
                    "batch represented in both splits."
                )

    scvi.model.SCVI.setup_anndata(train_adata, batch_key=batch_key)
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

    train_library = np.asarray(train_adata.X.sum(axis=1)).ravel()
    mean_log_library = float(np.log(np.maximum(train_library, 1.0)).mean())
    print(f"mean_log_library={mean_log_library:.4f} (exp={np.exp(mean_log_library):.1f})")

    n_batch = int(model.summary_stats["n_batch"])

    # Recover scVI's own category -> integer-index mapping so Stage 2 can encode
    # a cell's batch label the same way training did. Reading it back from the
    # fitted manager (rather than re-deriving it) guarantees the ordering matches.
    if batch_key is None:
        batch_category_to_index = None
    else:
        state = model.adata_manager.get_state_registry(scvi.REGISTRY_KEYS.BATCH_KEY)
        categories = [str(c) for c in state["categorical_mapping"]]
        batch_category_to_index = {c: i for i, c in enumerate(categories)}
        if len(categories) != n_batch:
            raise ValueError(
                f"scVI reports n_batch={n_batch} but the categorical mapping has "
                f"{len(categories)} entries; refusing to save an inconsistent checkpoint."
            )
        print(f"batch categories ({n_batch}): {categories[:8]}{' ...' if n_batch > 8 else ''}")

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
        "batch_key": batch_key,
        "batch_category_to_index": batch_category_to_index,
        # Stage 2 must feed the encoder the same units Stage 1 was trained on.
        "expression_transform": args.expression_transform,
        # ... and anything reading decode() back needs to know what comes out.
        "decode_units": decode_units,
        "gene_likelihood": args.gene_likelihood,
        "min_cells_detected": args.min_cells_detected,
        # Mean log library size over the training cells. The count path decodes
        # against a fixed 1e4 reference (px_scale is library-independent), but
        # the Normal path's px.loc IS library-scaled (px = Normal(px_rate, ...)),
        # so it needs the scale the model was actually fitted at.
        "mean_log_library": mean_log_library,
        "keep_technologies": sorted(keep_technologies) if keep_technologies else None,
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
