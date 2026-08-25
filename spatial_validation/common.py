from __future__ import annotations

import math
from typing import TYPE_CHECKING, Iterator, Sequence

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.spatial import cKDTree

from RNA_validation.common import GeneAligner, clean, transform_counts

if TYPE_CHECKING:
    import anndata as ad


def source_genes(adata: ad.AnnData) -> tuple[list[str], list[str]]:
    var = adata.var
    ids = var["gene_id"].astype(str).tolist() if "gene_id" in var else adata.var_names.astype(str).tolist()
    symbols = var["gene_symbol"].astype(str).tolist() if "gene_symbol" in var else adata.var_names.astype(str).tolist()
    ids = [clean(x) or str(name) for x, name in zip(ids, adata.var_names)]
    symbols = [clean(x) or str(name) for x, name in zip(symbols, adata.var_names)]
    return ids, symbols


def alignment_matrix(aligner: GeneAligner, adata: ad.AnnData):
    ids, symbols = source_genes(adata)
    result = aligner.resolve(ids, symbols)
    source_index = np.flatnonzero(result.source_to_model >= 0)
    target_index = result.source_to_model[source_index]
    mapping = sp.csr_matrix(
        (np.ones(len(source_index), dtype=np.float32), (source_index, target_index)),
        shape=(adata.n_vars, len(aligner.model)),
    )
    measured = np.zeros(len(aligner.model), dtype=bool)
    measured[np.unique(target_index)] = True
    return mapping, measured, result.report


def dense_rows(matrix) -> np.ndarray:
    if sp.issparse(matrix):
        return matrix.toarray().astype(np.float32, copy=False)
    return np.asarray(matrix, dtype=np.float32)


def aligned_batches(
    adata: ad.AnnData,
    mapping: sp.csr_matrix,
    transform: str,
    batch_size: int,
    max_observations: int = -1,
) -> Iterator[tuple[int, int, np.ndarray]]:
    n = adata.n_obs if max_observations < 0 else min(adata.n_obs, max_observations)
    for start in range(0, n, batch_size):
        stop = min(start + batch_size, n)
        source = adata.X[start:stop]
        library = np.asarray(source.sum(axis=1)).reshape(-1).astype(np.float64)
        aligned = source @ mapping
        aligned = dense_rows(aligned)
        yield start, stop, transform_counts(aligned, library, transform)


def coordinates(adata: ad.AnnData, n: int) -> np.ndarray:
    if "spatial" in adata.obsm:
        coords = np.asarray(adata.obsm["spatial"][:n, :2], dtype=np.float32)
    elif {"x_coordinate", "y_coordinate"}.issubset(adata.obs.columns):
        coords = adata.obs[["x_coordinate", "y_coordinate"]].iloc[:n].apply(pd.to_numeric, errors="coerce").to_numpy(np.float32)
    else:
        coords = np.full((n, 2), np.nan, dtype=np.float32)
    return coords


def morans_i(values: Sequence[float], coords: np.ndarray, k: int = 8) -> float:
    values = np.asarray(values, dtype=np.float64)
    coords = np.asarray(coords, dtype=np.float64)
    valid = np.isfinite(values) & np.isfinite(coords).all(axis=1)
    x = values[valid]
    xy = coords[valid]
    if len(x) < 4 or np.std(x) == 0:
        return math.nan
    neighbors = min(k + 1, len(x))
    indices = cKDTree(xy).query(xy, k=neighbors)[1]
    if indices.ndim == 1:
        return math.nan
    nbr = indices[:, 1:]
    centered = x - x.mean()
    numerator = (centered[:, None] * centered[nbr]).sum()
    denominator = (centered**2).sum()
    weight_sum = nbr.size
    return float(len(x) / weight_sum * numerator / denominator) if denominator > 0 else math.nan

