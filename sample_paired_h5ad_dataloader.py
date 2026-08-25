#!/usr/bin/env python3
"""
03_paired_h5ad_dataloader.py

PyTorch dataloader for paired MOSCOT h5ad shards.

Expected paired h5ad shard format
---------------------------------
Each shard is an AnnData object:

    adata.X                 = source / baseline expression
    adata.layers["target"]  = target / posttreatment expression
    adata.obs               = pair metadata, including probability
    adata.var               = unified gene metadata

Rows are paired training examples.
Columns are genes in the unified zero-filled Ensembl gene order.

Typical folder
--------------
    paired_training_h5ad_50k/
        paired_moscot_pairs.tsv
        paired_h5ad_manifest.tsv
        paired_h5ad_global_summary.tsv
        genes.tsv
        shards/
            paired_expr_shard_00000.h5ad
            paired_expr_shard_00001.h5ad
            ...

Training interpretation
-----------------------
    input  x_source = source / baseline gene expression
    output y_target = target / posttreatment gene expression
    weight          = MOSCOT transport probability

Important:
    Probability can be used in two ways:
        1. as loss weight: weighted loss = loss * probability
        2. as sampling weight: rows with larger probability are sampled more often

For first test training, recommended:
    sample_by_probability=False
    use probability as loss weight

Then later compare:
    sample_by_probability=True
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Tuple

import argparse
import math
import random
import re

import numpy as np
import pandas as pd
import scipy.sparse as sp
import anndata as ad

import torch
from torch.utils.data import IterableDataset, DataLoader


# =============================================================================
# Utilities
# =============================================================================

def to_dense_numpy(x) -> np.ndarray:
    """
    Convert AnnData matrix slice to dense float32 numpy array.
    """
    if sp.issparse(x):
        return x.toarray().astype(np.float32, copy=False)
    return np.asarray(x, dtype=np.float32)


def transform_expression_rows(x: np.ndarray, transform: str) -> np.ndarray:
    """
    Transform raw count rows without modifying the stored H5AD matrices.

    Dense counterpart of transform_expression_sparse() below. The two MUST
    produce identical values: Stage 1 transforms a pooled sparse matrix with
    the sparse version, while Stage 2 transforms dense row blocks with this
    one, and a frozen encoder is only valid if both stages feed it the same
    units. Keep them adjacent, and change them together.
    """
    x = np.asarray(x, dtype=np.float32)
    if transform == "none":
        return x
    if transform == "log1p_10k":
        library_size = x.sum(axis=1, dtype=np.float64, keepdims=True)
        scale = 10_000.0 / np.maximum(library_size, 1.0)
        return np.log1p(x * scale).astype(np.float32)
    raise ValueError(f"Unknown expression transform: {transform}")


def transform_expression_sparse(matrix: sp.spmatrix, transform: str) -> sp.csr_matrix:
    """
    Sparse counterpart of transform_expression_rows(); see that docstring.

    log1p(0) == 0, so applying it to the stored nonzero entries alone
    preserves sparsity exactly -- no densification of a 46k-gene matrix.

    Note on the zero-filled unified gene panel: the per-cell library is the sum
    over genes that dataset actually measured, since unmeasured genes are
    stored as exact zeros and contribute nothing. normalize_total is therefore
    well-defined here. The residual effect -- a cell measured on a narrower
    panel concentrates the same 1e4 over fewer genes -- is a technical
    difference between source studies, which is what the batch key corrects.
    """
    csr = matrix.tocsr().astype(np.float32)
    if transform == "none":
        return csr
    if transform == "log1p_10k":
        library = np.asarray(csr.sum(axis=1)).ravel()
        scale = (1e4 / np.maximum(library, 1.0)).astype(np.float32)
        scaled = (sp.diags(scale) @ csr).tocsr()
        scaled.data = np.log1p(scaled.data)
        return scaled
    raise ValueError(f"Unknown expression transform: {transform}")


def normalize_probability(prob: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    Convert raw probability values into a valid sampling distribution.
    """
    prob = np.asarray(prob, dtype=np.float64)
    prob = np.nan_to_num(prob, nan=0.0, posinf=0.0, neginf=0.0)
    prob = np.maximum(prob, 0.0)
    s = prob.sum()
    if s <= eps:
        return np.ones_like(prob, dtype=np.float64) / len(prob)
    return prob / s


def compute_group_to_fold(
    data_dir: str | Path,
    group_column: str,
    num_folds: int,
    seed: int,
) -> Dict[str, int]:
    """
    Deterministically assign each group (e.g. patient_id) to a fold.

    Shared by PairedH5ADBatchDataset and any standalone script (e.g. Stage-1
    scVI training) that must reproduce the identical train/val patient split
    for a given (group_column, num_folds, seed), so that different pipeline
    stages never see leakage between held-out patients.
    """
    pair_table = Path(data_dir) / "paired_moscot_pairs.tsv"
    if not pair_table.exists():
        raise FileNotFoundError(f"Grouped splitting requires {pair_table}")
    groups = pd.read_csv(
        pair_table, sep="\t", usecols=[group_column], dtype=str
    )[group_column].dropna().astype(str).unique()
    groups = np.asarray(sorted(groups), dtype=object)
    rng = np.random.default_rng(seed)
    rng.shuffle(groups)
    group_to_fold = {str(group): int(i % num_folds) for i, group in enumerate(groups)}
    if not group_to_fold:
        raise ValueError(f"No groups found for column {group_column!r}")
    return group_to_fold


# =============================================================================
# Batch / technology metadata
# =============================================================================

# Source-study accession embedded at the start of the patient directory name,
# e.g. "E-MTAB-12733-Pt1" -> "E-MTAB-12733", "GSE246613-P3" -> "GSE246613".
SOURCE_STUDY_PATTERN = re.compile(r"^([A-Z]-[A-Z]+-\d+|GSE\d+|SRP\d+|PRJ[A-Z]+\d+)")

DEFAULT_KEEP_TECHNOLOGIES = ("10x genomics",)


def derive_source_study(patient_id: str) -> str:
    """
    Extract the source-study accession from a patient identifier.

    The scCT-DB-derived patient directory names embed the GEO/ArrayExpress
    accession of the originating study. That accession is the correct
    *technical* batch unit: one study means one lab, one protocol, and one
    sequencing technology.

    This is deliberately preferred over dataset_id as a batch key. An scCT-DB
    dataset is partitioned by therapeutic regimen, cancer subtype, and drug
    response group -- biological variables that must NOT be corrected away as
    if they were technical batch. See docs/data_integration_decisions/README.md.

    Falls back to the full patient_id when no accession prefix is present, so
    the value is never empty.
    """
    text = str(patient_id).strip()
    match = SOURCE_STUDY_PATTERN.match(text)
    return match.group(1) if match else text


def normalize_technology(value: str) -> str:
    """
    Case- and whitespace-normalize a sequencing technology label.

    Required: the source metadata contains both "10X Genomics" and
    "10x Genomics" for the same platform. Without normalization these become
    two distinct batch categories and a model would try to correct a
    difference that does not exist.
    """
    return " ".join(str(value).strip().lower().split())


def load_technology_lookup(path: str | Path) -> Dict[str, str]:
    """
    Load a dataset_id -> normalized technology map.

    Expects a TSV with `dataset_id` and `technology` columns, as produced by
    scanning `uns["metadata"]` of the per-patient/dataset h5ads (the original
    Sample_info tables are no longer present on disk).
    """
    table = pd.read_csv(path, sep="\t", dtype=str)
    for column in ("dataset_id", "technology"):
        if column not in table.columns:
            raise ValueError(f"technology lookup {path} is missing column {column!r}")
    lookup: Dict[str, str] = {}
    for dataset_id, technology in zip(table["dataset_id"], table["technology"]):
        key = str(dataset_id).strip()
        value = normalize_technology(technology)
        previous = lookup.get(key)
        if previous is not None and previous != value:
            raise ValueError(
                f"dataset_id {key!r} maps to multiple technologies "
                f"({previous!r} and {value!r}); the lookup must be unambiguous."
            )
        lookup[key] = value
    if not lookup:
        raise ValueError(f"technology lookup {path} produced no entries")
    return lookup


def safe_get_gene_symbols(var: pd.DataFrame, gene_ids: List[str]) -> List[str]:
    """
    Return gene_symbol if present, otherwise gene_id.
    """
    if "gene_symbol" not in var.columns:
        return list(gene_ids)

    symbols = var["gene_symbol"].astype(str).tolist()
    out = []
    for gid, sym in zip(gene_ids, symbols):
        if sym == "" or sym.lower() in {"nan", "none", "null"}:
            out.append(gid)
        else:
            out.append(sym)
    return out


# =============================================================================
# Dataset
# =============================================================================

class PairedH5ADBatchDataset(IterableDataset):
    """
    Iterable PyTorch dataset that yields already-batched dictionaries.

    This is intentionally batch-level, not item-level, because h5ad sparse access
    is much more efficient when reading row blocks from a shard.

    Each yielded batch:
        {
            "x": source expression tensor, shape [B, G]
            "y": target expression tensor, shape [B, G]
            "weight": probability tensor, shape [B]
            "probability": same as weight, shape [B]
            "delta": elapsed-time tensor, shape [B]
            "pair_id": list[str]
            "patient_id": list[str]
            "dataset_id": list[str]
            "transition": list[str]
            "source_cell": list[str]
            "target_cell": list[str]
            "gene_ids": optional list[str], only if include_gene_info_in_batch=True
            "gene_symbols": optional list[str], only if include_gene_info_in_batch=True
        }

    Recommended DataLoader call:
        loader = DataLoader(dataset, batch_size=None, num_workers=0)

    Parameters
    ----------
    data_dir:
        Folder containing paired_h5ad_manifest.tsv and shards/.
    batch_size:
        Number of paired examples per yielded batch.
    shuffle_shards:
        Shuffle shard order each epoch.
    shuffle_rows:
        Shuffle rows within each shard each epoch.
    sample_by_probability:
        If True, rows within each shard are sampled with replacement using
        adata.obs[probability_col] as probability. This means high-probability
        pairs appear more often.
        If False, rows are sampled uniformly/shuffled once per epoch.
    use_probability_as_weight:
        If True, batch["weight"] = probability.
        If False, batch["weight"] = ones.
    probability_col:
        Column in adata.obs containing MOSCOT transport probability.
    time_delta_col:
        Optional column in adata.obs containing elapsed time between source and
        target. If None or missing, default_time_delta is used.
    default_time_delta:
        Elapsed-time value used when no time_delta_col is available.
    include_gene_info_in_batch:
        If True, include gene_ids and gene_symbols in every batch.
        This is convenient for debugging but slightly inefficient.
    dense:
        If True, return dense torch.float32 tensors.
        This is recommended for normal neural networks.
    split:
        One of "all", "train", or "val". Used with num_folds/fold_index for
        shard-level cross-validation.
    num_folds:
        Number of shard-level folds. Use 1 to disable folding.
    fold_index:
        Validation fold index in [0, num_folds).
    batch_key:
        Optional per-cell technical-batch label emitted as batch["batch_category"],
        for use as a conditioning covariate (e.g. a frozen scVI encoder trained
        with a batch key). One of:
            "source_study"  derived from patient_id via derive_source_study()
                            -- the recommended technical batch unit
            "technology"    normalized sequencing technology (requires
                            technology_lookup_path)
            "dataset_id"    NOT recommended: scCT-DB datasets are partitioned by
                            therapeutic regimen, cancer subtype, and drug
                            response group, so conditioning on it removes
                            biological signal
            "patient_id"    raw patient identifier
        None disables emission.
    technology_lookup_path:
        Optional TSV mapping dataset_id -> technology, as produced by scanning
        uns["metadata"] of the per-patient/dataset h5ads. Required for
        keep_technologies and for batch_key="technology".
    keep_technologies:
        Optional iterable of normalized technology names to retain; rows whose
        technology is not listed are dropped at load time. Defaults to None
        (keep everything). Pass DEFAULT_KEEP_TECHNOLOGIES to keep only 10x.
    """

    def __init__(
        self,
        data_dir: str | Path,
        batch_size: int = 128,
        shuffle_shards: bool = True,
        shuffle_rows: bool = True,
        sample_by_probability: bool = False,
        use_probability_as_weight: bool = True,
        probability_col: str = "probability",
        time_delta_col: Optional[str] = None,
        default_time_delta: float = 1.0,
        include_gene_info_in_batch: bool = False,
        dense: bool = True,
        expression_transform: str = "none",
        split: str = "all",
        num_folds: int = 1,
        fold_index: int = 0,
        group_column: Optional[str] = None,
        seed: int = 42,
        batch_key: Optional[str] = None,
        technology_lookup_path: Optional[str | Path] = None,
        keep_technologies: Optional[Iterable[str]] = None,
        allowed_batch_categories: Optional[Iterable[str]] = None,
        keep_gene_ids: Optional[Iterable[str]] = None,
    ):
        super().__init__()

        self.data_dir = Path(data_dir)
        self.batch_size = int(batch_size)
        self.shuffle_shards = bool(shuffle_shards)
        self.shuffle_rows = bool(shuffle_rows)
        self.sample_by_probability = bool(sample_by_probability)
        self.use_probability_as_weight = bool(use_probability_as_weight)
        self.probability_col = probability_col
        self.time_delta_col = time_delta_col
        self.default_time_delta = float(default_time_delta)
        self.include_gene_info_in_batch = bool(include_gene_info_in_batch)
        self.dense = bool(dense)
        self.expression_transform = str(expression_transform)
        self.split = split
        self.num_folds = int(num_folds)
        self.fold_index = int(fold_index)
        self.group_column = group_column
        self.seed = int(seed)
        self.batch_key = batch_key

        if self.expression_transform not in {"none", "log1p_10k"}:
            raise ValueError("expression_transform must be one of {'none', 'log1p_10k'}")

        valid_batch_keys = {None, "source_study", "technology", "dataset_id", "patient_id"}
        if self.batch_key not in valid_batch_keys:
            raise ValueError(f"batch_key must be one of {valid_batch_keys}, got {self.batch_key!r}")

        self.keep_technologies = (
            None
            if keep_technologies is None
            else {normalize_technology(t) for t in keep_technologies}
        )
        self.technology_lookup: Optional[Dict[str, str]] = None
        if technology_lookup_path is not None:
            self.technology_lookup = load_technology_lookup(technology_lookup_path)
        needs_lookup = self.keep_technologies is not None or self.batch_key == "technology"
        if needs_lookup and self.technology_lookup is None:
            raise ValueError(
                "technology_lookup_path is required when keep_technologies is set "
                "or batch_key='technology'."
            )

        # A frozen encoder can only embed batch categories it was trained on.
        # Rows outside that set are dropped here rather than raising deep in the
        # model, which is what happens when a study's patients all landed in the
        # held-out fold during Stage 1.
        self.allowed_batch_categories = (
            None if allowed_batch_categories is None else {str(c) for c in allowed_batch_categories}
        )
        if self.allowed_batch_categories is not None and self.batch_key is None:
            raise ValueError("allowed_batch_categories requires batch_key to be set")
        self.dropped_unknown_batch_rows = 0

        self.manifest_path = self.data_dir / "paired_h5ad_manifest.tsv"
        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Missing manifest: {self.manifest_path}")

        self.manifest = pd.read_csv(self.manifest_path, sep="\t")
        if self.manifest.empty:
            raise ValueError(f"Empty manifest: {self.manifest_path}")

        if "relative_shard_file" not in self.manifest.columns:
            raise ValueError("Manifest must contain relative_shard_file column.")

        if self.split not in {"all", "train", "val"}:
            raise ValueError("split must be one of {'all', 'train', 'val'}.")
        if self.num_folds <= 0:
            raise ValueError("num_folds must be positive.")
        if not 0 <= self.fold_index < self.num_folds:
            raise ValueError("fold_index must be in [0, num_folds).")
        if self.split != "all" and self.group_column is None and self.num_folds > len(self.manifest):
            raise ValueError(
                f"num_folds={self.num_folds} exceeds number of shards={len(self.manifest)}. "
                "Use fewer folds or create more shards."
            )

        self.group_to_fold: Dict[str, int] = {}
        if self.split != "all" and self.num_folds > 1 and self.group_column:
            self.group_to_fold = compute_group_to_fold(
                self.data_dir, self.group_column, self.num_folds, self.seed
            )
        elif self.split != "all" and self.num_folds > 1:
            shard_order = np.arange(len(self.manifest))
            rng = np.random.default_rng(self.seed)
            rng.shuffle(shard_order)
            val_positions = set(shard_order[self.fold_index::self.num_folds].tolist())
            keep_mask = [
                (i not in val_positions) if self.split == "train" else (i in val_positions)
                for i in range(len(self.manifest))
            ]
            self.manifest = self.manifest.loc[keep_mask].reset_index(drop=True)
            if self.manifest.empty:
                raise ValueError(
                    f"Empty {self.split} split for num_folds={self.num_folds}, "
                    f"fold_index={self.fold_index}."
                )

        self.shard_paths = [
            self.data_dir / rel
            for rel in self.manifest["relative_shard_file"].astype(str).tolist()
        ]

        for p in self.shard_paths:
            if not p.exists():
                raise FileNotFoundError(f"Missing shard: {p}")

        # Load gene metadata from first shard.
        first = ad.read_h5ad(self.shard_paths[0], backed="r")
        self.gene_var = first.var.copy()
        self.gene_ids = list(first.var_names.astype(str))
        self.gene_symbols = safe_get_gene_symbols(self.gene_var, self.gene_ids)
        first.file.close()

        # Optional gene subset. Applied as a column selection at read time, so
        # the shards on disk keep the full unified panel and different gene
        # sets can be compared without rebuilding data. Stage 2 takes this list
        # from the Stage-1 checkpoint, so the frozen encoder always receives
        # the exact columns, in the exact order, that it was fitted on.
        self.gene_index: Optional[np.ndarray] = None
        if keep_gene_ids is not None:
            wanted = [str(g) for g in keep_gene_ids]
            position = {g: i for i, g in enumerate(self.gene_ids)}
            missing = [g for g in wanted if g not in position]
            if missing:
                raise ValueError(
                    f"{len(missing)} requested gene(s) are absent from the shards, "
                    f"e.g. {missing[:5]}"
                )
            self.gene_index = np.asarray([position[g] for g in wanted], dtype=np.int64)
            self.gene_ids = wanted
            self.gene_var = self.gene_var.iloc[self.gene_index].copy()
            self.gene_symbols = safe_get_gene_symbols(self.gene_var, self.gene_ids)

        self.n_genes = len(self.gene_ids)

        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive.")

    def __len__(self) -> int:
        """
        Approximate number of batches per epoch.
        """
        n_pairs = int(pd.to_numeric(self.manifest["n_pairs"], errors="coerce").fillna(0).sum())
        return math.ceil(n_pairs / self.batch_size)

    def get_gene_info(self) -> pd.DataFrame:
        """
        Return gene metadata table with gene_id as first column.
        """
        out = self.gene_var.copy()
        out.insert(0, "gene_id", self.gene_ids)
        if "gene_symbol" not in out.columns:
            out.insert(1, "gene_symbol", self.gene_symbols)
        return out

    def _iter_shard_indices(self) -> List[int]:
        indices = list(range(len(self.shard_paths)))
        if self.shuffle_shards:
            rng = random.Random(self.seed + torch.initial_seed() % 10_000_000)
            rng.shuffle(indices)
        return indices

    def _row_technologies(self, obs: pd.DataFrame) -> np.ndarray:
        """Normalized technology per row, via the dataset_id lookup."""
        if self.technology_lookup is None:
            raise ValueError("technology lookup is not loaded")
        if "dataset_id" not in obs.columns:
            raise ValueError("shard obs has no dataset_id column; cannot resolve technology")
        return (
            obs["dataset_id"]
            .astype(str)
            .map(lambda d: self.technology_lookup.get(d.strip(), ""))
            .to_numpy()
        )

    def _technology_keep_mask(self, obs: pd.DataFrame) -> np.ndarray:
        """Boolean mask of rows whose technology is in keep_technologies."""
        technologies = self._row_technologies(obs)
        unknown = technologies == ""
        if unknown.any():
            missing = sorted(set(obs.loc[unknown, "dataset_id"].astype(str)))[:5]
            raise ValueError(
                f"{int(unknown.sum())} row(s) have a dataset_id absent from the "
                f"technology lookup, e.g. {missing}. Filtering on incomplete "
                "metadata would silently drop data."
            )
        return np.isin(technologies, list(self.keep_technologies))

    def _batch_categories(self, obs_batch: pd.DataFrame) -> Optional[List[str]]:
        if self.batch_key is None:
            return None
        if self.batch_key == "source_study":
            return [derive_source_study(p) for p in obs_batch["patient_id"].astype(str)]
        if self.batch_key == "technology":
            return list(self._row_technologies(obs_batch))
        return obs_batch[self.batch_key].astype(str).tolist()

    def _make_row_order(self, obs: pd.DataFrame, rng: np.random.Generator) -> np.ndarray:
        n = len(obs)

        if self.sample_by_probability:
            prob = pd.to_numeric(obs[self.probability_col], errors="coerce").fillna(0).to_numpy()
            p = normalize_probability(prob)
            # with replacement: high probability pairs appear more often
            return rng.choice(n, size=n, replace=True, p=p)

        order = np.arange(n)
        if self.shuffle_rows:
            rng.shuffle(order)
        return order

    def _read_batch(
        self,
        adata: ad.AnnData,
        obs: pd.DataFrame,
        row_idx: np.ndarray,
    ) -> Dict:
        # Gene subset is applied BEFORE the transform, so normalize_total's
        # library is the sum over the genes actually being modelled. Stage 1
        # does the same, in the same order; reversing it in either place would
        # give the two stages different library sizes for the same cell.
        x_raw = to_dense_numpy(adata.X[row_idx, :])
        y_raw = to_dense_numpy(adata.layers["target"][row_idx, :])
        if self.gene_index is not None:
            x_raw = x_raw[:, self.gene_index]
            y_raw = y_raw[:, self.gene_index]
        x_np = transform_expression_rows(x_raw, self.expression_transform)
        y_np = transform_expression_rows(y_raw, self.expression_transform)

        x = torch.from_numpy(x_np)
        y = torch.from_numpy(y_np)

        prob_np = pd.to_numeric(obs.iloc[row_idx][self.probability_col], errors="coerce").fillna(0).to_numpy(np.float32)
        probability = torch.from_numpy(prob_np)

        if self.time_delta_col and self.time_delta_col in obs.columns:
            delta_np = pd.to_numeric(obs.iloc[row_idx][self.time_delta_col], errors="coerce").fillna(
                self.default_time_delta
            ).to_numpy(np.float32)
        else:
            delta_np = np.full(len(row_idx), self.default_time_delta, dtype=np.float32)
        delta = torch.from_numpy(delta_np)

        if self.use_probability_as_weight:
            weight = probability.clone()
        else:
            weight = torch.ones_like(probability)

        obs_batch = obs.iloc[row_idx]

        batch = {
            "x": x,
            "y": y,
            "weight": weight,
            "probability": probability,
            "delta": delta,
            "pair_id": obs_batch["pair_id"].astype(str).tolist() if "pair_id" in obs_batch.columns else None,
            "patient_id": obs_batch["patient_id"].astype(str).tolist() if "patient_id" in obs_batch.columns else None,
            "dataset_id": obs_batch["dataset_id"].astype(str).tolist() if "dataset_id" in obs_batch.columns else None,
            "transition": obs_batch["transition"].astype(str).tolist() if "transition" in obs_batch.columns else None,
            "source_cell": obs_batch["source_cell_original_obs"].astype(str).tolist() if "source_cell_original_obs" in obs_batch.columns else None,
            "target_cell": obs_batch["target_cell_original_obs"].astype(str).tolist() if "target_cell_original_obs" in obs_batch.columns else None,
        }

        batch["batch_category"] = self._batch_categories(obs_batch)

        if self.include_gene_info_in_batch:
            batch["gene_ids"] = self.gene_ids
            batch["gene_symbols"] = self.gene_symbols

        return batch

    def __iter__(self) -> Iterator[Dict]:
        worker_info = torch.utils.data.get_worker_info()
        worker_id = 0 if worker_info is None else worker_info.id
        num_workers = 1 if worker_info is None else worker_info.num_workers

        # Split shards across workers if num_workers > 1.
        shard_indices = self._iter_shard_indices()
        shard_indices = shard_indices[worker_id::num_workers]

        rng = np.random.default_rng(self.seed + worker_id + int(torch.initial_seed() % 10_000_000))

        for shard_idx in shard_indices:
            shard_path = self.shard_paths[shard_idx]

            adata = ad.read_h5ad(shard_path, backed="r")

            if "target" not in adata.layers:
                adata.file.close()
                raise ValueError(f"Missing target layer in shard: {shard_path}")

            if self.probability_col not in adata.obs.columns:
                adata.file.close()
                raise ValueError(f"Missing probability column '{self.probability_col}' in shard: {shard_path}")

            obs = adata.obs.copy()

            # Combine every row-level filter into one eligibility mask so that
            # technology filtering and fold splitting compose correctly.
            eligible_mask = np.ones(len(obs), dtype=bool)

            if self.keep_technologies is not None:
                try:
                    eligible_mask &= self._technology_keep_mask(obs)
                except ValueError:
                    adata.file.close()
                    raise

            if self.allowed_batch_categories is not None:
                categories = np.asarray(self._batch_categories(obs), dtype=object)
                known = np.array(
                    [c in self.allowed_batch_categories for c in categories], dtype=bool
                )
                self.dropped_unknown_batch_rows += int((~known & eligible_mask).sum())
                eligible_mask &= known

            if self.group_to_fold:
                if self.group_column not in obs.columns:
                    adata.file.close()
                    raise ValueError(f"Missing group column {self.group_column!r} in {shard_path}")
                folds = obs[self.group_column].astype(str).map(self.group_to_fold)
                if folds.isna().any():
                    adata.file.close()
                    raise ValueError(f"Unassigned groups in {shard_path}")
                fold_values = folds.to_numpy()
                eligible_mask &= (
                    (fold_values != self.fold_index)
                    if self.split == "train"
                    else (fold_values == self.fold_index)
                )

            if eligible_mask.all():
                order = self._make_row_order(obs, rng)
            else:
                eligible = np.flatnonzero(eligible_mask)
                if len(eligible) == 0:
                    adata.file.close()
                    continue
                local_order = self._make_row_order(obs.iloc[eligible], rng)
                order = eligible[local_order]

            for start in range(0, len(order), self.batch_size):
                end = min(start + self.batch_size, len(order))
                row_idx = order[start:end]
                yield self._read_batch(adata, obs, row_idx)

            adata.file.close()


# =============================================================================
# Convenience builder
# =============================================================================

def build_paired_h5ad_loader(
    data_dir: str | Path,
    batch_size: int = 128,
    shuffle_shards: bool = True,
    shuffle_rows: bool = True,
    sample_by_probability: bool = False,
    use_probability_as_weight: bool = True,
    include_gene_info_in_batch: bool = False,
    time_delta_col: Optional[str] = None,
    default_time_delta: float = 1.0,
    expression_transform: str = "none",
    split: str = "all",
    num_folds: int = 1,
    fold_index: int = 0,
    group_column: Optional[str] = None,
    num_workers: int = 0,
    seed: int = 42,
    batch_key: Optional[str] = None,
    technology_lookup_path: Optional[str | Path] = None,
    keep_technologies: Optional[Iterable[str]] = None,
    allowed_batch_categories: Optional[Iterable[str]] = None,
    keep_gene_ids: Optional[Iterable[str]] = None,
) -> Tuple[PairedH5ADBatchDataset, DataLoader]:
    """
    Build dataset and DataLoader.

    Use batch_size=None in DataLoader because the dataset already yields batches.
    """
    dataset = PairedH5ADBatchDataset(
        data_dir=data_dir,
        batch_size=batch_size,
        shuffle_shards=shuffle_shards,
        shuffle_rows=shuffle_rows,
        sample_by_probability=sample_by_probability,
        use_probability_as_weight=use_probability_as_weight,
        time_delta_col=time_delta_col,
        default_time_delta=default_time_delta,
        expression_transform=expression_transform,
        split=split,
        num_folds=num_folds,
        fold_index=fold_index,
        group_column=group_column,
        include_gene_info_in_batch=include_gene_info_in_batch,
        seed=seed,
        batch_key=batch_key,
        technology_lookup_path=technology_lookup_path,
        keep_technologies=keep_technologies,
        allowed_batch_categories=allowed_batch_categories,
        keep_gene_ids=keep_gene_ids,
    )

    loader = DataLoader(
        dataset,
        batch_size=None,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    return dataset, loader


# =============================================================================
# Example model/loss usage
# =============================================================================

def weighted_mse_loss(pred: torch.Tensor, target: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """
    Weighted MSE over cells and genes.

    pred:   [B, G]
    target: [B, G]
    weight: [B]

    This makes each pair weight apply to the whole gene-expression vector.
    """
    per_pair_loss = ((pred - target) ** 2).mean(dim=1)
    return (per_pair_loss * weight).sum() / (weight.sum() + 1e-8)


def example_train_loop(data_dir: str | Path, batch_size: int = 64):
    """
    Minimal example. Replace the model with your real transition model.
    """
    dataset, loader = build_paired_h5ad_loader(
        data_dir=data_dir,
        batch_size=batch_size,
        shuffle_shards=True,
        shuffle_rows=True,
        sample_by_probability=False,
        use_probability_as_weight=True,
        num_workers=0,
        seed=42,
    )

    print("Number of genes:", dataset.n_genes)
    print("First 10 genes:", dataset.gene_ids[:10])
    print("First 10 symbols:", dataset.gene_symbols[:10])
    print("Approx batches per epoch:", len(dataset))

    # Dummy model: linear gene-to-gene mapping.
    # For real training, replace this with VAE / MLP / transformer / perturbation model.
    model = torch.nn.Sequential(
        torch.nn.Linear(dataset.n_genes, 512),
        torch.nn.ReLU(),
        torch.nn.Linear(512, dataset.n_genes),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    model.train()
    for step, batch in enumerate(loader):
        x = batch["x"].to(device)
        y = batch["y"].to(device)
        w = batch["weight"].to(device)

        pred = model(x)
        loss = weighted_mse_loss(pred, y, w)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 10 == 0:
            print(
                f"step={step} loss={loss.item():.6f} "
                f"x={tuple(x.shape)} y={tuple(y.shape)} "
                f"weight_mean={w.mean().item():.4f}"
            )

        # small smoke test
        if step >= 20:
            break


# =============================================================================
# CLI smoke test
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("paired_training_h5ad_50k"))
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--sample-by-probability", action="store_true")
    parser.add_argument("--include-gene-info-in-batch", action="store_true")
    parser.add_argument("--time-delta-col", type=str, default=None)
    parser.add_argument("--default-time-delta", type=float, default=1.0)
    parser.add_argument("--train-smoke-test", action="store_true")
    args = parser.parse_args()

    dataset, loader = build_paired_h5ad_loader(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        shuffle_shards=True,
        shuffle_rows=True,
        sample_by_probability=args.sample_by_probability,
        use_probability_as_weight=True,
        time_delta_col=args.time_delta_col,
        default_time_delta=args.default_time_delta,
        include_gene_info_in_batch=args.include_gene_info_in_batch,
        num_workers=args.num_workers,
        seed=42,
    )

    print("=" * 80)
    print("Paired h5ad dataloader smoke test")
    print("=" * 80)
    print("data_dir:", args.data_dir)
    print("n_genes:", dataset.n_genes)
    print("first 10 gene_ids:", dataset.gene_ids[:10])
    print("first 10 gene_symbols:", dataset.gene_symbols[:10])
    print("approx batches per epoch:", len(dataset))
    print("sample_by_probability:", args.sample_by_probability)

    gene_info = dataset.get_gene_info()
    print("\nGene metadata columns:")
    print(list(gene_info.columns))
    print("\nGene metadata head:")
    print(gene_info.head(10).to_string(index=False))

    batch = next(iter(loader))

    print("\nFirst batch:")
    print("x shape:", tuple(batch["x"].shape))
    print("y shape:", tuple(batch["y"].shape))
    print("weight shape:", tuple(batch["weight"].shape))
    print("delta shape:", tuple(batch["delta"].shape))
    print("delta min/mean/max:",
          float(batch["delta"].min()),
          float(batch["delta"].mean()),
          float(batch["delta"].max()))
    print("probability min/mean/max:",
          float(batch["probability"].min()),
          float(batch["probability"].mean()),
          float(batch["probability"].max()))
    print("first pair ids:", batch["pair_id"][:3])
    print("first patient ids:", batch["patient_id"][:3])
    print("first transitions:", batch["transition"][:3])

    if args.train_smoke_test:
        print("\nRunning minimal train smoke test...")
        example_train_loop(args.data_dir, args.batch_size)


if __name__ == "__main__":
    main()
