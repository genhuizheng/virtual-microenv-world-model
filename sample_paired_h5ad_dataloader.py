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
from typing import Dict, Iterator, List, Optional, Tuple

import argparse
import math
import random

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
    """Transform raw count rows without modifying the stored H5AD matrices."""
    x = np.asarray(x, dtype=np.float32)
    if transform == "none":
        return x
    if transform == "log1p_10k":
        library_size = x.sum(axis=1, dtype=np.float64, keepdims=True)
        scale = 10_000.0 / np.maximum(library_size, 1.0)
        return np.log1p(x * scale).astype(np.float32)
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

        if self.expression_transform not in {"none", "log1p_10k"}:
            raise ValueError("expression_transform must be one of {'none', 'log1p_10k'}")

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
        self.n_genes = len(self.gene_ids)
        first.file.close()

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
        x_np = transform_expression_rows(
            to_dense_numpy(adata.X[row_idx, :]), self.expression_transform
        )
        y_np = transform_expression_rows(
            to_dense_numpy(adata.layers["target"][row_idx, :]), self.expression_transform
        )

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
            if self.group_to_fold:
                if self.group_column not in obs.columns:
                    adata.file.close()
                    raise ValueError(f"Missing group column {self.group_column!r} in {shard_path}")
                folds = obs[self.group_column].astype(str).map(self.group_to_fold)
                if folds.isna().any():
                    adata.file.close()
                    raise ValueError(f"Unassigned groups in {shard_path}")
                eligible = np.flatnonzero(
                    (folds.to_numpy() != self.fold_index)
                    if self.split == "train"
                    else (folds.to_numpy() == self.fold_index)
                )
                if len(eligible) == 0:
                    adata.file.close()
                    continue
                local_order = self._make_row_order(obs.iloc[eligible], rng)
                order = eligible[local_order]
            else:
                order = self._make_row_order(obs, rng)

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
