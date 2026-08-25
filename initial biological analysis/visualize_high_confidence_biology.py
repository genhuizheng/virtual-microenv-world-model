#!/usr/bin/env python3
"""
Global rough biology analysis from high-confidence OT pairs.

Inputs:
    batch_processed_unified_genes_zero_fill/<patient>/<dataset>/
      baseline.h5ad
      posttreatment*.h5ad
      OT_high_confidence/<metric>/ot_confidence_*.csv

Outputs:
    <base-dir>/initial_biological_analysis_global/
      pair_file_summary.csv
      cell_annotation_transitions.csv
      cell_annotation_fraction_by_metric.csv
      gene_level_global_<metric>.csv
      top_genes_global_<metric>.csv
      plots/*.png

This script performs paired pseudobulk-style pre/post testing by tumor type:
high-confidence OT pairs define matched source/target cells, each
patient/dataset/transition is treated as a paired pseudobulk replicate, genes
are tested with paired t-tests across replicates, and BH FDR is reported.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
import argparse
import gc
import re

import numpy as np
import pandas as pd

ad = None
sp = None


DEFAULT_BASE_DIR = Path("batch_processed_unified_genes_zero_fill")
DEFAULT_HIGH_CONF_DIR = "OT_high_confidence"
DEFAULT_OUT_DIR = "initial_biological_analysis_global"
EPS = 1e-6

TUMOR_TYPE_CANDIDATES = [
    "cancer_subtype",
    "cancer_doid",
    "tumor_type",
    "tumour_type",
    "cancer_type",
    "cancer",
    "disease",
    "disease_type",
    "diagnosis",
    "primary_diagnosis",
    "study",
    "cohort",
    "project",
    "tissue",
    "tissue_type",
    "organ",
    "site",
    "primary_site",
    "oncotree_code",
    "histology",
    "histological_type",
]

ANNOTATION_CANDIDATES = [
    "cell_type",
    "celltype",
    "celltypes",
    "cell_type_major",
    "cell_type_fine",
    "major_cell_type",
    "minor_cell_type",
    "annotation",
    "annotations",
    "manual_annotation",
    "predicted_cell_type",
    "celltypist_cell_label",
    "singler_label",
    "leiden",
    "louvain",
    "cluster",
    "seurat_clusters",
]

DEFAULT_SIGNATURES = {
    "cytotoxic_t_cell": ["GZMB", "GZMA", "PRF1", "NKG7", "GNLY", "IFNG", "CTSW"],
    "t_cell_exhaustion": ["PDCD1", "CTLA4", "LAG3", "HAVCR2", "TIGIT", "TOX", "ENTPD1"],
    "interferon_response": ["ISG15", "IFIT1", "IFIT2", "IFIT3", "MX1", "OAS1", "STAT1", "IRF7"],
    "antigen_presentation": ["HLA-A", "HLA-B", "HLA-C", "B2M", "TAP1", "TAP2", "HLA-DRA", "HLA-DRB1"],
    "cell_cycle_proliferation": ["MKI67", "TOP2A", "PCNA", "MCM2", "MCM5", "STMN1", "TYMS"],
    "apoptosis": ["BAX", "BAK1", "CASP3", "CASP8", "FAS", "FASLG", "BBC3", "PMAIP1"],
    "hypoxia": ["HIF1A", "VEGFA", "CA9", "SLC2A1", "LDHA", "ENO1", "PGK1"],
    "emt_invasion": ["VIM", "FN1", "SNAI1", "SNAI2", "ZEB1", "ZEB2", "MMP2", "MMP9"],
    "myeloid_inflammation": ["LYZ", "S100A8", "S100A9", "IL1B", "TNF", "CXCL8", "FCN1"],
    "immune_checkpoint": ["CD274", "PDCD1LG2", "PDCD1", "CTLA4", "LAG3", "TIGIT", "HAVCR2"],
    "tgfb_signature": ["TGFB1", "TGFBR1", "TGFBR2", "SMAD2", "SMAD3", "SERPINE1", "COL1A1"],
}


def require_h5ad_stack() -> None:
    global ad, sp
    if ad is None:
        import anndata as _ad

        ad = _ad
    if sp is None:
        import scipy.sparse as _sp

        sp = _sp


def require_plot_stack():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def clean_str(x) -> str:
    if pd.isna(x):
        return ""
    s = str(x).strip()
    if s.lower() in {"", "nan", "none", "null", "na"}:
        return ""
    return s


def strip_moscot_concat_suffix(cell_id) -> str:
    return re.sub(r"_[01]$", "", clean_str(cell_id))


def infer_target_h5ad_from_transition(transition: str) -> str:
    transition = clean_str(transition)
    if not transition.startswith("baseline_to_"):
        return ""
    target_label = transition.replace("baseline_to_", "", 1)
    if target_label == "posttreatment":
        return "posttreatment.h5ad"
    return f"{target_label}.h5ad"


def discover_confidence_csvs(base_dir: Path, high_conf_dir_name: str, metrics: Optional[List[str]]) -> List[Path]:
    paths = sorted(base_dir.glob(f"*/*/{high_conf_dir_name}/*/ot_confidence_*.csv"))
    paths = [
        p
        for p in paths
        if "summary" not in p.name.lower()
        and not p.name.lower().endswith("_summary.csv")
    ]
    if metrics:
        metric_set = set(metrics)
        paths = [p for p in paths if p.parent.name in metric_set]
    if not paths:
        raise FileNotFoundError(f"No high-confidence pair CSVs found under {base_dir}/{high_conf_dir_name}")
    return paths


def infer_context(path: Path, base_dir: Path, high_conf_dir_name: str) -> Dict:
    rel = path.relative_to(base_dir)
    parts = rel.parts
    patient_id = parts[0]
    dataset_id = parts[1]
    metric = parts[3] if len(parts) > 3 and parts[2] == high_conf_dir_name else path.parent.name
    dataset_dir = base_dir / patient_id / dataset_id

    stem = path.stem.replace("ot_confidence_", "", 1)
    suffix = f"_{metric}"
    if stem.endswith(suffix):
        stem = stem[: -len(suffix)]
    if stem.endswith("_source_to_target"):
        transition = stem[: -len("_source_to_target")]
        pair_direction = "source_to_target"
    elif stem.endswith("_target_to_source"):
        transition = stem[: -len("_target_to_source")]
        pair_direction = "target_to_source"
    else:
        transition = ""
        pair_direction = ""

    return {
        "patient_id": patient_id,
        "dataset_id": dataset_id,
        "metric": metric,
        "dataset_dir": dataset_dir,
        "transition": transition,
        "pair_direction": pair_direction,
        "confidence_csv": path,
        "baseline_h5ad": dataset_dir / "baseline.h5ad",
        "target_h5ad": dataset_dir / infer_target_h5ad_from_transition(transition),
    }


def filter_pairs(df: pd.DataFrame, top_fraction: Optional[float], min_confidence: Optional[float]) -> pd.DataFrame:
    out = df.copy()
    if "confidence_score" not in out.columns:
        raise ValueError("High-confidence CSV missing confidence_score column.")
    out["confidence_score"] = pd.to_numeric(out["confidence_score"], errors="coerce")
    out = out[out["confidence_score"].notna()].copy()
    if min_confidence is not None:
        out = out[out["confidence_score"] >= float(min_confidence)].copy()
    if top_fraction is not None:
        if not 0.0 < top_fraction <= 1.0:
            raise ValueError("--top-fraction must be in (0, 1].")
        threshold = out["confidence_score"].quantile(1.0 - top_fraction)
        out = out[out["confidence_score"] >= threshold].copy()
    out = out.sort_values("confidence_score", ascending=False)
    return out.reset_index(drop=True)


def choose_annotation_column(adata_obj, preferred: Optional[str]) -> Optional[str]:
    if preferred and preferred in adata_obj.obs.columns:
        return preferred
    lower_to_real = {c.lower(): c for c in adata_obj.obs.columns}
    for c in ANNOTATION_CANDIDATES:
        if c.lower() in lower_to_real:
            return lower_to_real[c.lower()]
    return None


def choose_metadata_column(adata_obj, preferred: Optional[str], candidates: List[str]) -> Optional[str]:
    if preferred and preferred in adata_obj.obs.columns:
        return preferred
    lower_to_real = {c.lower(): c for c in adata_obj.obs.columns}
    for c in candidates:
        if c.lower() in lower_to_real:
            return lower_to_real[c.lower()]
    return None


def dominant_obs_value(adata_obj, col: Optional[str]) -> str:
    if not col or col not in adata_obj.obs.columns:
        return ""
    vals = adata_obj.obs[col].astype(str)
    vals = vals[~vals.str.lower().isin(["", "nan", "none", "null", "na"])]
    if vals.empty:
        return ""
    return str(vals.value_counts().index[0])


def metadata_value(adata_obj, keys: List[str]) -> Tuple[str, str]:
    meta = adata_obj.uns.get("metadata", {})
    if not isinstance(meta, dict):
        return "", ""
    lower_to_real = {str(k).lower(): k for k in meta.keys()}
    for key in keys:
        real = lower_to_real.get(key.lower())
        if real is None:
            continue
        value = clean_str(meta.get(real, ""))
        if value:
            return value, str(real)
    return "", ""


def infer_tumor_type(baseline, target, ctx: Dict, preferred: Optional[str]) -> Tuple[str, str]:
    if preferred:
        value, key = metadata_value(baseline, [preferred])
        if value:
            return value, f"uns.metadata.{key}"
        value, key = metadata_value(target, [preferred])
        if value:
            return value, f"uns.metadata.{key}"

    value, key = metadata_value(baseline, ["cancer_subtype", "cancer_doid"])
    if value:
        return value, f"uns.metadata.{key}"
    value, key = metadata_value(target, ["cancer_subtype", "cancer_doid"])
    if value:
        return value, f"uns.metadata.{key}"

    col = choose_metadata_column(baseline, preferred, TUMOR_TYPE_CANDIDATES)
    value = dominant_obs_value(baseline, col)
    if value:
        return value, col or ""
    col = choose_metadata_column(target, preferred, TUMOR_TYPE_CANDIDATES)
    value = dominant_obs_value(target, col)
    if value:
        return value, col or ""
    return f"{ctx['patient_id']}__{ctx['dataset_id']}", "fallback_patient_dataset"


def index_cells(adata_obj, cell_ids: Iterable[str], label: str) -> np.ndarray:
    mapping = {}

    def add_values(values) -> None:
        for i, value in enumerate(values):
            raw = clean_str(value)
            if not raw:
                continue
            mapping.setdefault(raw, i)
            mapping.setdefault(strip_moscot_concat_suffix(raw), i)

    add_values(adata_obj.obs_names.astype(str))
    for col in ("raw_barcode", "barcode"):
        if col in adata_obj.obs.columns:
            add_values(adata_obj.obs[col].astype(str).values)

    idx_values = []
    missing = []
    for cell_id in cell_ids:
        key = clean_str(cell_id)
        match = mapping.get(key)
        if match is None:
            match = mapping.get(strip_moscot_concat_suffix(key))
        if match is None:
            missing.append(key)
            idx_values.append(-1)
        else:
            idx_values.append(match)
    idx = pd.Series(idx_values, index=pd.Index(list(cell_ids), dtype=str))
    if missing:
        raise KeyError(f"Missing {label} cells in h5ad; examples={missing[:10]}")
    return idx.astype(int).to_numpy()


def to_csr(x):
    require_h5ad_stack()
    if sp.issparse(x):
        return x.tocsr()
    return sp.csr_matrix(x)


def get_gene_table(adata_obj) -> pd.DataFrame:
    out = adata_obj.var.copy()
    out.insert(0, "gene_id", adata_obj.var_names.astype(str))
    if "gene_name" not in out.columns:
        out["gene_name"] = adata_obj.var_names.astype(str)
    return out.reset_index(drop=True)


def align_genes(baseline, target):
    baseline_names = pd.Index(baseline.var_names.astype(str))
    target_names = pd.Index(target.var_names.astype(str))
    if len(baseline_names) == len(target_names) and np.array_equal(baseline_names, target_names):
        return baseline, target
    common = baseline_names.intersection(target_names)
    if len(common) == 0:
        raise ValueError("No shared genes between baseline and target h5ad.")
    return baseline[:, common].copy(), target[:, common].copy()


def init_gene_accumulator(gene_df: pd.DataFrame) -> Dict:
    n = len(gene_df)
    return {
        "gene_df": gene_df.copy(),
        "n_pairs": 0,
        "source_sum": np.zeros(n, dtype=np.float64),
        "target_sum": np.zeros(n, dtype=np.float64),
        "source_nnz": np.zeros(n, dtype=np.float64),
        "target_nnz": np.zeros(n, dtype=np.float64),
        "replicate_source_means": [],
        "replicate_target_means": [],
        "replicate_ids": [],
        "replicate_n_pairs": [],
    }


def add_expression_to_accumulator(acc: Dict, pairs: pd.DataFrame, baseline, target, chunk_size: int, replicate_id: str) -> None:
    source_cells = pairs["source_cell_original_obs"].astype(str).map(strip_moscot_concat_suffix).tolist()
    target_cells = pairs["target_cell_original_obs"].astype(str).map(strip_moscot_concat_suffix).tolist()
    rep_source_sum = np.zeros_like(acc["source_sum"])
    rep_target_sum = np.zeros_like(acc["target_sum"])
    for start in range(0, len(pairs), chunk_size):
        end = min(start + chunk_size, len(pairs))
        src_idx = index_cells(baseline, source_cells[start:end], "source")
        tgt_idx = index_cells(target, target_cells[start:end], "target")
        X_src = to_csr(baseline.X[src_idx, :])
        X_tgt = to_csr(target.X[tgt_idx, :])
        src_sum = np.asarray(X_src.sum(axis=0)).ravel()
        tgt_sum = np.asarray(X_tgt.sum(axis=0)).ravel()
        acc["source_sum"] += src_sum
        acc["target_sum"] += tgt_sum
        rep_source_sum += src_sum
        rep_target_sum += tgt_sum
        acc["source_nnz"] += np.asarray((X_src > 0).sum(axis=0)).ravel()
        acc["target_nnz"] += np.asarray((X_tgt > 0).sum(axis=0)).ravel()
        acc["n_pairs"] += end - start
        del X_src, X_tgt
        gc.collect()
    n_pairs = max(len(pairs), 1)
    acc["replicate_source_means"].append(rep_source_sum / n_pairs)
    acc["replicate_target_means"].append(rep_target_sum / n_pairs)
    acc["replicate_ids"].append(replicate_id)
    acc["replicate_n_pairs"].append(int(len(pairs)))


def merge_accumulator(target: Dict, source: Dict) -> None:
    target["n_pairs"] += int(source["n_pairs"])
    target["source_sum"] += source["source_sum"]
    target["target_sum"] += source["target_sum"]
    target["source_nnz"] += source["source_nnz"]
    target["target_nnz"] += source["target_nnz"]
    target["replicate_source_means"].extend(source["replicate_source_means"])
    target["replicate_target_means"].extend(source["replicate_target_means"])
    target["replicate_ids"].extend(source["replicate_ids"])
    target["replicate_n_pairs"].extend(source["replicate_n_pairs"])


def merge_accumulator_dict(target: Dict[str, Dict], source: Dict[str, Dict]) -> None:
    for key, src_acc in source.items():
        if key not in target:
            target[key] = src_acc
        else:
            merge_accumulator(target[key], src_acc)


def finalize_gene_accumulator(acc: Dict) -> pd.DataFrame:
    n_pairs = max(int(acc["n_pairs"]), 1)
    out = acc["gene_df"].copy()
    source_mean = acc["source_sum"] / n_pairs
    target_mean = acc["target_sum"] / n_pairs
    delta = target_mean - source_mean
    out["n_pairs"] = int(acc["n_pairs"])
    out["source_mean_expr"] = source_mean
    out["target_mean_expr"] = target_mean
    out["mean_delta_target_minus_source"] = delta
    out["log2fc_target_vs_source"] = np.log2((target_mean + EPS) / (source_mean + EPS))
    out["source_pct_expr"] = acc["source_nnz"] / n_pairs
    out["target_pct_expr"] = acc["target_nnz"] / n_pairs
    out["pct_expr_delta"] = out["target_pct_expr"] - out["source_pct_expr"]
    out["abs_mean_delta"] = np.abs(delta)
    return out.sort_values(["abs_mean_delta", "target_pct_expr"], ascending=[False, False]).reset_index(drop=True)


def bh_fdr(p_values: np.ndarray) -> np.ndarray:
    p = np.asarray(p_values, dtype=float)
    out = np.full_like(p, np.nan, dtype=float)
    valid = np.isfinite(p)
    if valid.sum() == 0:
        return out
    pv = p[valid]
    order = np.argsort(pv)
    ranked = pv[order]
    n = len(ranked)
    q = ranked * n / (np.arange(n) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.clip(q, 0, 1)
    tmp = np.empty_like(q)
    tmp[order] = q
    out[valid] = tmp
    return out


def paired_pseudobulk_de(acc: Dict) -> pd.DataFrame:
    gene_df = finalize_gene_accumulator(acc)
    src = np.vstack(acc["replicate_source_means"]) if acc["replicate_source_means"] else np.empty((0, len(gene_df)))
    tgt = np.vstack(acc["replicate_target_means"]) if acc["replicate_target_means"] else np.empty((0, len(gene_df)))
    n_rep = src.shape[0]
    p_values = np.full(len(gene_df), np.nan, dtype=float)
    t_stats = np.full(len(gene_df), np.nan, dtype=float)
    mean_pseudobulk_source = np.full(len(gene_df), np.nan, dtype=float)
    mean_pseudobulk_target = np.full(len(gene_df), np.nan, dtype=float)
    mean_paired_delta = np.full(len(gene_df), np.nan, dtype=float)

    if n_rep > 0:
        mean_pseudobulk_source = src.mean(axis=0)
        mean_pseudobulk_target = tgt.mean(axis=0)
        delta = tgt - src
        mean_paired_delta = delta.mean(axis=0)

    if n_rep >= 2:
        from scipy import stats

        for j in range(src.shape[1]):
            d = tgt[:, j] - src[:, j]
            if np.allclose(d, d[0]):
                p_values[j] = 1.0
                t_stats[j] = 0.0
            else:
                res = stats.ttest_rel(tgt[:, j], src[:, j], nan_policy="omit")
                p_values[j] = float(res.pvalue) if np.isfinite(res.pvalue) else np.nan
                t_stats[j] = float(res.statistic) if np.isfinite(res.statistic) else np.nan

    gene_df["n_replicates"] = int(n_rep)
    gene_df["pseudobulk_source_mean"] = mean_pseudobulk_source
    gene_df["pseudobulk_target_mean"] = mean_pseudobulk_target
    gene_df["pseudobulk_paired_delta"] = mean_paired_delta
    gene_df["pseudobulk_log2fc"] = np.log2((mean_pseudobulk_target + EPS) / (mean_pseudobulk_source + EPS))
    gene_df["paired_t_stat"] = t_stats
    gene_df["paired_t_pvalue"] = p_values
    gene_df["paired_t_fdr_bh"] = bh_fdr(p_values)
    gene_df["significant_fdr_0p05"] = gene_df["paired_t_fdr_bh"] <= 0.05
    gene_df["significant_fdr_0p10"] = gene_df["paired_t_fdr_bh"] <= 0.10
    return gene_df.sort_values(["paired_t_fdr_bh", "abs_mean_delta"], ascending=[True, False], na_position="last").reset_index(drop=True)


def annotation_transition_table(pairs: pd.DataFrame, baseline, target, source_col: Optional[str], target_col: Optional[str], ctx: Dict) -> pd.DataFrame:
    if source_col is None and target_col is None:
        return pd.DataFrame()
    source_cells = pairs["source_cell_original_obs"].astype(str).map(strip_moscot_concat_suffix)
    target_cells = pairs["target_cell_original_obs"].astype(str).map(strip_moscot_concat_suffix)
    src_idx = index_cells(baseline, source_cells.tolist(), "source")
    tgt_idx = index_cells(target, target_cells.tolist(), "target")
    src_labels = baseline.obs.iloc[src_idx][source_col].astype(str).to_numpy() if source_col else np.array(["unknown"] * len(pairs))
    tgt_labels = target.obs.iloc[tgt_idx][target_col].astype(str).to_numpy() if target_col else np.array(["unknown"] * len(pairs))
    tab = pd.DataFrame({"source_annotation": src_labels, "target_annotation": tgt_labels})
    tab = tab.value_counts(["source_annotation", "target_annotation"]).reset_index(name="n_pairs")
    tab["fraction"] = tab["n_pairs"] / max(tab["n_pairs"].sum(), 1)
    for key in ["patient_id", "dataset_id", "metric", "transition", "pair_direction", "tumor_type"]:
        tab[key] = ctx[key]
    tab["source_annotation_column"] = source_col or ""
    tab["target_annotation_column"] = target_col or ""
    return tab


def annotation_fraction_replicates(pairs: pd.DataFrame, baseline, target, source_col: Optional[str], target_col: Optional[str], ctx: Dict) -> pd.DataFrame:
    if source_col is None and target_col is None:
        return pd.DataFrame()
    source_cells = pairs["source_cell_original_obs"].astype(str).map(strip_moscot_concat_suffix)
    target_cells = pairs["target_cell_original_obs"].astype(str).map(strip_moscot_concat_suffix)
    src_idx = index_cells(baseline, source_cells.tolist(), "source")
    tgt_idx = index_cells(target, target_cells.tolist(), "target")
    src_labels = baseline.obs.iloc[src_idx][source_col].astype(str).to_numpy() if source_col else np.array(["unknown"] * len(pairs))
    tgt_labels = target.obs.iloc[tgt_idx][target_col].astype(str).to_numpy() if target_col else np.array(["unknown"] * len(pairs))
    labels = sorted(set(src_labels.tolist()) | set(tgt_labels.tolist()))
    rows = []
    n = max(len(pairs), 1)
    replicate_id = f"{ctx['patient_id']}|{ctx['dataset_id']}|{ctx['transition']}|{ctx['pair_direction']}"
    for label in labels:
        rows.append(
            {
                "patient_id": ctx["patient_id"],
                "dataset_id": ctx["dataset_id"],
                "tumor_type": ctx["tumor_type"],
                "metric": ctx["metric"],
                "transition": ctx["transition"],
                "pair_direction": ctx["pair_direction"],
                "replicate_id": replicate_id,
                "cell_annotation": label,
                "source_fraction": float((src_labels == label).sum() / n),
                "target_fraction": float((tgt_labels == label).sum() / n),
                "fraction_delta_target_minus_source": float((tgt_labels == label).sum() / n - (src_labels == label).sum() / n),
                "n_pairs": int(len(pairs)),
                "source_annotation_column": source_col or "",
                "target_annotation_column": target_col or "",
            }
        )
    return pd.DataFrame(rows)


def plot_top_genes(gene_df: pd.DataFrame, out_png: Path, title: str, top_n: int) -> None:
    plt = require_plot_stack()
    top = gene_df.reindex(gene_df["mean_delta_target_minus_source"].abs().sort_values(ascending=False).index).head(top_n)
    labels = top["gene_name"].fillna(top["gene_id"]).astype(str).tolist()
    values = top["mean_delta_target_minus_source"].to_numpy()
    colors = ["#b83232" if v > 0 else "#2f5d9b" for v in values]
    fig_h = max(4.0, 0.25 * len(top) + 1.0)
    fig, ax = plt.subplots(figsize=(8, fig_h))
    ax.barh(np.arange(len(top)), values, color=colors)
    ax.set_yticks(np.arange(len(top)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Mean expression delta: post - pre")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def plot_annotation_fractions(annotation_df: pd.DataFrame, out_png: Path, title: str, top_n: int) -> None:
    if annotation_df.empty:
        return
    plt = require_plot_stack()
    top = annotation_df.groupby("target_annotation", as_index=False)["n_pairs"].sum()
    top = top.sort_values("n_pairs", ascending=False).head(top_n)
    top["fraction"] = top["n_pairs"] / max(top["n_pairs"].sum(), 1)
    fig, ax = plt.subplots(figsize=(8, max(4, 0.28 * len(top) + 1)))
    ax.barh(np.arange(len(top)), top["fraction"], color="#4f7f6f")
    ax.set_yticks(np.arange(len(top)))
    ax.set_yticklabels(top["target_annotation"].astype(str), fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Fraction among shown target annotations")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def paired_cell_fraction_tests(frac_df: pd.DataFrame) -> pd.DataFrame:
    if frac_df.empty:
        return pd.DataFrame()
    from scipy import stats

    rows = []
    group_cols = ["tumor_type", "metric", "cell_annotation"]
    for keys, group in frac_df.groupby(group_cols, sort=False):
        tumor_type, metric, cell_annotation = keys
        n_rep = group["replicate_id"].nunique()
        deltas = group["fraction_delta_target_minus_source"].to_numpy(dtype=float)
        p_value = np.nan
        t_stat = np.nan
        if n_rep >= 2 and not np.allclose(deltas, deltas[0]):
            res = stats.ttest_1samp(deltas, popmean=0.0, nan_policy="omit")
            p_value = float(res.pvalue) if np.isfinite(res.pvalue) else np.nan
            t_stat = float(res.statistic) if np.isfinite(res.statistic) else np.nan
        elif n_rep >= 2:
            p_value = 1.0
            t_stat = 0.0
        rows.append(
            {
                "tumor_type": tumor_type,
                "metric": metric,
                "cell_annotation": cell_annotation,
                "n_replicates": int(n_rep),
                "mean_source_fraction": float(group["source_fraction"].mean()),
                "mean_target_fraction": float(group["target_fraction"].mean()),
                "mean_fraction_delta_target_minus_source": float(np.nanmean(deltas)),
                "paired_t_stat": t_stat,
                "paired_t_pvalue": p_value,
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["paired_t_fdr_bh"] = out.groupby(["tumor_type", "metric"], group_keys=False)["paired_t_pvalue"].transform(lambda x: bh_fdr(x.to_numpy()))
    return out.sort_values(["paired_t_fdr_bh", "mean_fraction_delta_target_minus_source"], ascending=[True, False], na_position="last")


def transition_enrichment_tests(annotation_df: pd.DataFrame) -> pd.DataFrame:
    if annotation_df.empty:
        return pd.DataFrame()
    from scipy import stats

    rows = []
    group_cols = ["tumor_type", "metric", "source_annotation", "target_annotation"]
    totals = annotation_df.groupby(["tumor_type", "metric"], as_index=False)["n_pairs"].sum().rename(columns={"n_pairs": "total_pairs"})
    merged = annotation_df.merge(totals, on=["tumor_type", "metric"], how="left")
    for keys, group in merged.groupby(group_cols, sort=False):
        tumor_type, metric, src, tgt = keys
        observed = int(group["n_pairs"].sum())
        total = int(group["total_pairs"].iloc[0])
        source_total = int(merged[(merged["tumor_type"] == tumor_type) & (merged["metric"] == metric) & (merged["source_annotation"] == src)]["n_pairs"].sum())
        target_total = int(merged[(merged["tumor_type"] == tumor_type) & (merged["metric"] == metric) & (merged["target_annotation"] == tgt)]["n_pairs"].sum())
        expected = source_total * target_total / max(total, 1)
        table = np.array(
            [
                [observed, max(source_total - observed, 0)],
                [max(target_total - observed, 0), max(total - source_total - target_total + observed, 0)],
            ],
            dtype=float,
        )
        try:
            odds_ratio, p_value = stats.fisher_exact(table)
        except Exception:
            odds_ratio, p_value = np.nan, np.nan
        rows.append(
            {
                "tumor_type": tumor_type,
                "metric": metric,
                "source_annotation": src,
                "target_annotation": tgt,
                "observed_pairs": observed,
                "expected_pairs_independence": expected,
                "observed_over_expected": observed / (expected + EPS),
                "odds_ratio": odds_ratio,
                "fisher_pvalue": p_value,
                "total_pairs": total,
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["fisher_fdr_bh"] = out.groupby(["tumor_type", "metric"], group_keys=False)["fisher_pvalue"].transform(lambda x: bh_fdr(x.to_numpy()))
    return out.sort_values(["fisher_fdr_bh", "observed_over_expected"], ascending=[True, False], na_position="last")


def plot_volcano(de_df: pd.DataFrame, out_png: Path, title: str, fdr_cutoff: float, top_label_n: int = 12) -> None:
    plt = require_plot_stack()
    df = de_df.copy()
    df["neg_log10_fdr"] = -np.log10(df["paired_t_fdr_bh"].clip(lower=1e-300))
    sig = df["paired_t_fdr_bh"] <= fdr_cutoff
    fig, ax = plt.subplots(figsize=(7.5, 6.0))
    ax.scatter(df.loc[~sig, "pseudobulk_log2fc"], df.loc[~sig, "neg_log10_fdr"], s=8, color="#9aa0a6", alpha=0.55)
    ax.scatter(df.loc[sig, "pseudobulk_log2fc"], df.loc[sig, "neg_log10_fdr"], s=10, color="#b83232", alpha=0.75)
    ax.axhline(-np.log10(fdr_cutoff), color="black", linewidth=0.8, linestyle="--")
    ax.axvline(0, color="black", linewidth=0.6)
    label_df = df.sort_values(["paired_t_fdr_bh", "abs_mean_delta"], ascending=[True, False]).head(top_label_n)
    for _, row in label_df.iterrows():
        label = clean_str(row.get("gene_name", "")) or clean_str(row.get("gene_id", ""))
        if label:
            ax.text(row["pseudobulk_log2fc"], row["neg_log10_fdr"], label, fontsize=7)
    ax.set_xlabel("Paired pseudobulk log2FC: post vs pre")
    ax.set_ylabel("-log10(BH FDR)")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_png, dpi=220)
    plt.close(fig)


def plot_replicate_delta_heatmap(acc: Dict, de_df: pd.DataFrame, out_png: Path, title: str, top_n: int) -> None:
    if not acc["replicate_source_means"] or len(acc["replicate_source_means"]) < 2:
        return
    plt = require_plot_stack()
    src = np.vstack(acc["replicate_source_means"])
    tgt = np.vstack(acc["replicate_target_means"])
    delta = tgt - src
    top = de_df.sort_values(["paired_t_fdr_bh", "abs_mean_delta"], ascending=[True, False], na_position="last").head(top_n)
    if top.empty:
        return
    idx = top.index.to_numpy()
    mat = delta[:, idx]
    mat = mat - np.nanmean(mat, axis=0, keepdims=True)
    scale = np.nanstd(mat, axis=0, keepdims=True)
    mat = mat / np.where(scale == 0, 1.0, scale)
    fig, ax = plt.subplots(figsize=(max(8, 0.28 * len(idx)), max(4.5, 0.25 * mat.shape[0] + 1.5)))
    im = ax.imshow(mat, aspect="auto", cmap="coolwarm", vmin=-2.5, vmax=2.5)
    labels = top["gene_name"].fillna(top["gene_id"]).astype(str).tolist()
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=90, fontsize=7)
    ax.set_yticks(np.arange(len(acc["replicate_ids"])))
    ax.set_yticklabels(acc["replicate_ids"], fontsize=6)
    ax.set_title(title)
    ax.set_xlabel("Top DE genes")
    ax.set_ylabel("Paired pseudobulk replicate")
    fig.colorbar(im, ax=ax, shrink=0.75, label="z-scored post-pre delta")
    fig.tight_layout()
    fig.savefig(out_png, dpi=220)
    plt.close(fig)


def read_gmt(path: Path, min_size: int, max_size: int) -> Dict[str, set]:
    gene_sets = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            name = parts[0]
            genes = {g.strip().upper() for g in parts[2:] if g.strip()}
            if min_size <= len(genes) <= max_size:
                gene_sets[name] = genes
    if not gene_sets:
        raise ValueError(f"No gene sets found after size filtering: {path}")
    return gene_sets


def preranked_gsea(de_df: pd.DataFrame, gene_sets: Dict[str, set], permutations: int, seed: int, min_overlap: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    genes = de_df["gene_name"].fillna(de_df["gene_id"]).astype(str).str.upper().to_numpy()
    scores = de_df["pseudobulk_log2fc"].fillna(0).to_numpy(dtype=float)
    order = np.argsort(-scores)
    genes = genes[order]
    scores = scores[order]
    abs_scores = np.abs(scores)
    gene_to_idx = {g: i for i, g in enumerate(genes)}
    rows = []

    def enrichment_score(hit_idx: np.ndarray) -> float:
        hit = np.zeros(len(genes), dtype=bool)
        hit[hit_idx] = True
        n_hit = hit.sum()
        n_miss = len(hit) - n_hit
        if n_hit == 0 or n_miss == 0:
            return 0.0
        hit_weights = abs_scores * hit
        hit_norm = hit_weights.sum()
        if hit_norm <= 0:
            hit_weights = hit.astype(float)
            hit_norm = hit_weights.sum()
        running = np.cumsum(hit_weights / hit_norm - (~hit).astype(float) / n_miss)
        max_es = running.max()
        min_es = running.min()
        return float(max_es if abs(max_es) >= abs(min_es) else min_es)

    for name, gene_set in gene_sets.items():
        hit_idx = np.array([gene_to_idx[g] for g in gene_set if g in gene_to_idx], dtype=int)
        if len(hit_idx) < min_overlap:
            continue
        observed = enrichment_score(hit_idx)
        null = np.zeros(permutations, dtype=float)
        for i in range(permutations):
            random_idx = rng.choice(len(genes), size=len(hit_idx), replace=False)
            null[i] = enrichment_score(random_idx)
        if observed >= 0:
            denom = np.mean(null[null >= 0]) if np.any(null >= 0) else 1.0
            p_value = (np.sum(null >= observed) + 1) / (permutations + 1)
        else:
            denom = abs(np.mean(null[null < 0])) if np.any(null < 0) else 1.0
            p_value = (np.sum(null <= observed) + 1) / (permutations + 1)
        rows.append(
            {
                "pathway": name,
                "overlap": int(len(hit_idx)),
                "enrichment_score": observed,
                "normalized_enrichment_score": observed / (denom + EPS),
                "pvalue": float(p_value),
                "leading_edge_genes": ";".join(genes[np.sort(hit_idx)[:50]].tolist()),
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["fdr_bh"] = bh_fdr(out["pvalue"].to_numpy())
    return out.sort_values(["fdr_bh", "normalized_enrichment_score"], ascending=[True, False], na_position="last")


def plot_gsea_bar(gsea_df: pd.DataFrame, out_png: Path, title: str, top_n: int = 20) -> None:
    if gsea_df.empty:
        return
    plt = require_plot_stack()
    top = gsea_df.sort_values("fdr_bh").head(top_n).copy()
    labels = top["pathway"].astype(str).str.slice(0, 70).tolist()
    values = top["normalized_enrichment_score"].to_numpy()
    colors = ["#b83232" if v > 0 else "#2f5d9b" for v in values]
    fig, ax = plt.subplots(figsize=(9, max(4.5, 0.32 * len(top) + 1.0)))
    ax.barh(np.arange(len(top)), values, color=colors)
    ax.set_yticks(np.arange(len(top)))
    ax.set_yticklabels(labels, fontsize=7)
    ax.invert_yaxis()
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Normalized enrichment score")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_png, dpi=220)
    plt.close(fig)


def run_enrichr_public(de_df: pd.DataFrame, libraries: List[str], fdr_cutoff: float, out_prefix: Path) -> List[Path]:
    sig = de_df[
        (de_df["paired_t_fdr_bh"] <= fdr_cutoff)
        & (de_df["gene_name"].notna() | de_df["gene_id"].notna())
    ].copy()
    if sig.empty:
        return []
    sig = sig.sort_values(["paired_t_fdr_bh", "abs_mean_delta"], ascending=[True, False]).head(500)
    genes = sig["gene_name"].fillna(sig["gene_id"]).astype(str).dropna().unique().tolist()
    if len(genes) < 5:
        return []
    try:
        import gseapy as gp
    except Exception as exc:
        print(f"WARNING: gseapy not available; skipping Enrichr public enrichment: {exc}")
        return []

    written = []
    for lib in libraries:
        try:
            enr = gp.enrichr(gene_list=genes, gene_sets=lib, organism="Human", outdir=None, cutoff=1.0)
            res = enr.results.copy()
            out_csv = out_prefix.with_name(out_prefix.name + f"_enrichr_{re.sub(r'[^A-Za-z0-9_.-]+', '_', lib)}.csv")
            res.to_csv(out_csv, index=False)
            written.append(out_csv)
        except Exception as exc:
            print(f"WARNING: Enrichr library failed ({lib}): {exc}")
    return written


def confidence_metric_robustness(de_results: Dict[Tuple[str, str, str], pd.DataFrame], fdr_cutoff: float, top_n: int = 200) -> pd.DataFrame:
    rows = []
    by_scope_tumor = {}
    for (scope, tumor_type, metric), df in de_results.items():
        by_scope_tumor.setdefault((scope, tumor_type), {})[metric] = df
    for (scope, tumor_type), metric_map in by_scope_tumor.items():
        metrics = sorted(metric_map)
        for i, m1 in enumerate(metrics):
            for m2 in metrics[i + 1:]:
                d1 = metric_map[m1]
                d2 = metric_map[m2]
                genes1 = set(d1[d1["paired_t_fdr_bh"] <= fdr_cutoff]["gene_name"].fillna(d1["gene_id"]).astype(str))
                genes2 = set(d2[d2["paired_t_fdr_bh"] <= fdr_cutoff]["gene_name"].fillna(d2["gene_id"]).astype(str))
                top1 = set(d1.head(top_n)["gene_name"].fillna(d1.head(top_n)["gene_id"]).astype(str))
                top2 = set(d2.head(top_n)["gene_name"].fillna(d2.head(top_n)["gene_id"]).astype(str))
                rows.append(
                    {
                        "scope": scope,
                        "tumor_type": tumor_type,
                        "metric_a": m1,
                        "metric_b": m2,
                        "n_sig_a": len(genes1),
                        "n_sig_b": len(genes2),
                        "n_sig_overlap": len(genes1 & genes2),
                        "sig_jaccard": len(genes1 & genes2) / max(len(genes1 | genes2), 1),
                        "top_n": int(top_n),
                        "top_overlap": len(top1 & top2),
                        "top_jaccard": len(top1 & top2) / max(len(top1 | top2), 1),
                    }
                )
    return pd.DataFrame(rows)


def plot_cell_fraction_delta_tests(frac_test: pd.DataFrame, out_png: Path, title: str, top_n: int) -> None:
    if frac_test.empty:
        return
    plt = require_plot_stack()
    top = frac_test.reindex(frac_test["mean_fraction_delta_target_minus_source"].abs().sort_values(ascending=False).index).head(top_n)
    labels = top["cell_annotation"].astype(str).tolist()
    values = top["mean_fraction_delta_target_minus_source"].to_numpy()
    colors = ["#b83232" if v > 0 else "#2f5d9b" for v in values]
    fig, ax = plt.subplots(figsize=(8, max(4.0, 0.3 * len(top) + 1.0)))
    ax.barh(np.arange(len(top)), values, color=colors)
    ax.set_yticks(np.arange(len(top)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Mean paired fraction delta: post - pre")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_png, dpi=220)
    plt.close(fig)


def signature_indices(gene_df: pd.DataFrame, signatures: Dict[str, List[str]]) -> Dict[str, np.ndarray]:
    gene_symbols = gene_df["gene_name"].fillna(gene_df["gene_id"]).astype(str).str.upper().to_numpy()
    gene_to_idx = {}
    for i, g in enumerate(gene_symbols):
        gene_to_idx.setdefault(g, i)
    out = {}
    for name, genes in signatures.items():
        idx = [gene_to_idx[g.upper()] for g in genes if g.upper() in gene_to_idx]
        if idx:
            out[name] = np.array(sorted(set(idx)), dtype=int)
    return out


def paired_signature_tests(acc: Dict, signatures: Dict[str, List[str]]) -> pd.DataFrame:
    if not acc["replicate_source_means"] or len(acc["replicate_source_means"]) == 0:
        return pd.DataFrame()
    from scipy import stats

    src = np.vstack(acc["replicate_source_means"])
    tgt = np.vstack(acc["replicate_target_means"])
    idx_map = signature_indices(acc["gene_df"], signatures)
    rows = []
    for sig, idx in idx_map.items():
        src_score = src[:, idx].mean(axis=1)
        tgt_score = tgt[:, idx].mean(axis=1)
        delta = tgt_score - src_score
        n_rep = len(delta)
        p_value = np.nan
        t_stat = np.nan
        if n_rep >= 2 and not np.allclose(delta, delta[0]):
            res = stats.ttest_rel(tgt_score, src_score, nan_policy="omit")
            p_value = float(res.pvalue) if np.isfinite(res.pvalue) else np.nan
            t_stat = float(res.statistic) if np.isfinite(res.statistic) else np.nan
        elif n_rep >= 2:
            p_value = 1.0
            t_stat = 0.0
        rows.append(
            {
                "scope": acc.get("scope", ""),
                "tumor_type": acc.get("tumor_type", ""),
                "cell_type": acc.get("cell_type", ""),
                "metric": acc.get("metric", ""),
                "signature": sig,
                "n_signature_genes_found": int(len(idx)),
                "n_replicates": int(n_rep),
                "mean_source_score": float(np.nanmean(src_score)),
                "mean_target_score": float(np.nanmean(tgt_score)),
                "mean_delta_target_minus_source": float(np.nanmean(delta)),
                "paired_t_stat": t_stat,
                "paired_t_pvalue": p_value,
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["paired_t_fdr_bh"] = out.groupby(["scope", "tumor_type", "cell_type", "metric"], group_keys=False)["paired_t_pvalue"].transform(lambda x: bh_fdr(x.to_numpy()))
    return out.sort_values(["paired_t_fdr_bh", "mean_delta_target_minus_source"], ascending=[True, False], na_position="last")


def plot_signature_deltas(sig_df: pd.DataFrame, out_png: Path, title: str) -> None:
    if sig_df.empty:
        return
    plt = require_plot_stack()
    df = sig_df.sort_values("mean_delta_target_minus_source")
    values = df["mean_delta_target_minus_source"].to_numpy()
    colors = ["#b83232" if v > 0 else "#2f5d9b" for v in values]
    fig, ax = plt.subplots(figsize=(8, max(4.0, 0.35 * len(df) + 1.0)))
    ax.barh(np.arange(len(df)), values, color=colors)
    ax.set_yticks(np.arange(len(df)))
    ax.set_yticklabels(df["signature"].astype(str), fontsize=8)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Mean paired signature delta: post - pre")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_png, dpi=220)
    plt.close(fig)


def plot_transition_heatmap(annotation_df: pd.DataFrame, out_png: Path, title: str, top_n: int) -> None:
    if annotation_df.empty:
        return
    plt = require_plot_stack()
    mat = annotation_df.pivot_table(
        index="source_annotation",
        columns="target_annotation",
        values="n_pairs",
        aggfunc="sum",
        fill_value=0,
    )
    top_rows = mat.sum(axis=1).sort_values(ascending=False).head(top_n).index
    top_cols = mat.sum(axis=0).sort_values(ascending=False).head(top_n).index
    mat = mat.loc[top_rows, top_cols]
    denom = mat.to_numpy().sum()
    values = mat.to_numpy(dtype=float) / max(denom, 1)
    fig, ax = plt.subplots(figsize=(max(6, 0.35 * mat.shape[1] + 2), max(5, 0.35 * mat.shape[0] + 2)))
    im = ax.imshow(values, aspect="auto", cmap="viridis")
    ax.set_xticks(np.arange(mat.shape[1]))
    ax.set_xticklabels(mat.columns.astype(str), rotation=90, fontsize=7)
    ax.set_yticks(np.arange(mat.shape[0]))
    ax.set_yticklabels(mat.index.astype(str), fontsize=7)
    ax.set_xlabel("Post-treatment target annotation")
    ax.set_ylabel("Pre-treatment source annotation")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, shrink=0.75, label="fraction of high-confidence pairs")
    fig.tight_layout()
    fig.savefig(out_png, dpi=220)
    plt.close(fig)


def accumulator_key(metric: str, tumor_type: str, scope: str) -> str:
    return f"{scope}::{metric}::{tumor_type}"


def add_cell_type_specific_accumulators(
    accumulators: Dict[str, Dict],
    metric: str,
    tumor_type: str,
    pairs: pd.DataFrame,
    baseline,
    target,
    source_col: Optional[str],
    target_col: Optional[str],
    ctx: Dict,
    args: argparse.Namespace,
    replicate_id: str,
) -> None:
    if not args.cell_type_de:
        return
    axis = args.cell_type_de_axis
    if axis == "target" and target_col is None:
        return
    if axis == "source" and source_col is None:
        return
    source_cells = pairs["source_cell_original_obs"].astype(str).map(strip_moscot_concat_suffix)
    target_cells = pairs["target_cell_original_obs"].astype(str).map(strip_moscot_concat_suffix)
    if axis == "target":
        idx = index_cells(target, target_cells.tolist(), "target")
        labels = target.obs.iloc[idx][target_col].astype(str).to_numpy()
    else:
        idx = index_cells(baseline, source_cells.tolist(), "source")
        labels = baseline.obs.iloc[idx][source_col].astype(str).to_numpy()

    tmp = pairs.copy()
    tmp["__cell_type_de_label"] = labels
    for label, sub in tmp.groupby("__cell_type_de_label", sort=False):
        if len(sub) < args.min_pairs_per_cell_type_de:
            continue
        safe_label = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(label))[:80]
        scope = f"cell_type_{axis}"
        key = f"{scope}::{metric}::{tumor_type}::{safe_label}"
        if key not in accumulators:
            accumulators[key] = init_gene_accumulator(get_gene_table(baseline))
            accumulators[key]["metric"] = metric
            accumulators[key]["tumor_type"] = tumor_type
            accumulators[key]["cell_type"] = str(label)
            accumulators[key]["scope"] = scope
        add_expression_to_accumulator(accumulators[key], sub.drop(columns=["__cell_type_de_label"]), baseline, target, args.chunk_size, replicate_id)


def process_one_confidence_csv(path: Path, args: argparse.Namespace, accumulators: Dict[str, Dict]) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    require_h5ad_stack()
    ctx = infer_context(path, args.base_dir, args.high_confidence_dir_name)
    pairs = pd.read_csv(path)
    required = {"source_cell_original_obs", "target_cell_original_obs", "confidence_score"}
    missing = required - set(pairs.columns)
    if missing:
        print(f"WARNING: skipping non-pair CSV with missing columns {missing}: {path}")
        return pd.DataFrame(), pd.DataFrame(), {
            **ctx,
            "confidence_csv": str(path),
            "n_pairs_used": 0,
            "mean_confidence_score": np.nan,
            "skipped_reason": f"missing_columns:{','.join(sorted(missing))}",
        }
    pairs = filter_pairs(pairs, args.top_fraction, args.min_confidence)
    if pairs.empty:
        return pd.DataFrame(), pd.DataFrame(), {**ctx, "n_pairs_used": 0, "mean_confidence_score": np.nan}

    baseline = ad.read_h5ad(ctx["baseline_h5ad"])
    target = ad.read_h5ad(ctx["target_h5ad"])
    baseline, target = align_genes(baseline, target)
    tumor_type, tumor_type_column = infer_tumor_type(baseline, target, ctx, args.tumor_type_col)
    ctx["tumor_type"] = tumor_type
    ctx["tumor_type_column"] = tumor_type_column

    metric = ctx["metric"]
    replicate_id = f"{ctx['patient_id']}|{ctx['dataset_id']}|{ctx['transition']}|{ctx['pair_direction']}"
    tumor_key = accumulator_key(metric, tumor_type, "tumor_type")
    if tumor_key not in accumulators:
        accumulators[tumor_key] = init_gene_accumulator(get_gene_table(baseline))
        accumulators[tumor_key]["metric"] = metric
        accumulators[tumor_key]["tumor_type"] = tumor_type
        accumulators[tumor_key]["scope"] = "tumor_type"
    add_expression_to_accumulator(accumulators[tumor_key], pairs, baseline, target, args.chunk_size, replicate_id)

    if args.include_global:
        global_key = accumulator_key(metric, "all", "global")
        if global_key not in accumulators:
            accumulators[global_key] = init_gene_accumulator(get_gene_table(baseline))
            accumulators[global_key]["metric"] = metric
            accumulators[global_key]["tumor_type"] = "all"
            accumulators[global_key]["scope"] = "global"
        add_expression_to_accumulator(accumulators[global_key], pairs, baseline, target, args.chunk_size, replicate_id)

    source_col = choose_annotation_column(baseline, args.source_annotation_col)
    target_col = choose_annotation_column(target, args.target_annotation_col)
    annot = annotation_transition_table(pairs, baseline, target, source_col, target_col, ctx)
    frac_rep = annotation_fraction_replicates(pairs, baseline, target, source_col, target_col, ctx)
    add_cell_type_specific_accumulators(
        accumulators,
        metric,
        tumor_type,
        pairs,
        baseline,
        target,
        source_col,
        target_col,
        ctx,
        args,
        replicate_id,
    )

    summary = {
        **ctx,
        "tumor_type": tumor_type,
        "tumor_type_column": tumor_type_column,
        "confidence_csv": str(path),
        "n_pairs_used": int(len(pairs)),
        "mean_confidence_score": float(pairs["confidence_score"].mean()),
        "median_confidence_score": float(pairs["confidence_score"].median()),
        "source_annotation_column": source_col or "",
        "target_annotation_column": target_col or "",
    }
    del baseline, target
    gc.collect()
    return annot, frac_rep, summary


def process_one_confidence_csv_worker(task) -> Tuple[pd.DataFrame, pd.DataFrame, Dict, Dict[str, Dict]]:
    path_str, args = task
    local_accumulators: Dict[str, Dict] = {}
    annot, frac_rep, summary = process_one_confidence_csv(Path(path_str), args, local_accumulators)
    return annot, frac_rep, summary, local_accumulators


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir", type=Path, default=DEFAULT_BASE_DIR)
    parser.add_argument("--high-confidence-dir-name", type=str, default=DEFAULT_HIGH_CONF_DIR)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--metrics", type=str, default="combined_concentration")
    parser.add_argument("--top-fraction", type=float, default=0.25)
    parser.add_argument("--min-confidence", type=float, default=None)
    parser.add_argument("--chunk-size", type=int, default=2048)
    parser.add_argument("--tumor-type-col", type=str, default=None)
    parser.add_argument("--source-annotation-col", type=str, default=None)
    parser.add_argument("--target-annotation-col", type=str, default=None)
    parser.add_argument("--include-global", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fdr-cutoff", type=float, default=0.05)
    parser.add_argument("--gmt-file", type=Path, default=None)
    parser.add_argument("--enrichr-libraries", type=str, default="")
    parser.add_argument("--gsea-permutations", type=int, default=1000)
    parser.add_argument("--gsea-min-size", type=int, default=10)
    parser.add_argument("--gsea-max-size", type=int, default=500)
    parser.add_argument("--gsea-min-overlap", type=int, default=10)
    parser.add_argument("--gsea-seed", type=int, default=0)
    parser.add_argument("--cell-type-de", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--cell-type-de-axis", type=str, default="target", choices=["source", "target"])
    parser.add_argument("--min-pairs-per-cell-type-de", type=int, default=200)
    parser.add_argument("--module-score-analysis", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--transition-heatmaps", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--robustness-report", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--robustness-top-n", type=int, default=200)
    parser.add_argument("--top-genes-plot", type=int, default=30)
    parser.add_argument("--top-annotations-plot", type=int, default=25)
    parser.add_argument("--limit-files", type=int, default=None)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--worker-chunksize", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir or (args.base_dir / DEFAULT_OUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    metrics = None if args.metrics.strip().lower() == "all" else [m.strip() for m in args.metrics.split(",") if m.strip()]
    paths = discover_confidence_csvs(args.base_dir, args.high_confidence_dir_name, metrics)
    if args.limit_files is not None:
        paths = paths[: args.limit_files]
    gene_sets = None
    if args.gmt_file is not None:
        gene_sets = read_gmt(args.gmt_file, args.gsea_min_size, args.gsea_max_size)
    enrichr_libraries = [x.strip() for x in args.enrichr_libraries.split(",") if x.strip()]

    print("=" * 80)
    print("Global high-confidence OT biological visualization")
    print("=" * 80)
    print(f"Base dir: {args.base_dir}")
    print(f"Out dir:  {out_dir}")
    print(f"Metrics:  {args.metrics}")
    print(f"Tumor type column: {args.tumor_type_col or 'auto'}")
    print("Pair cap: none; using all pairs after top-fraction/min-confidence filters")
    print(f"GSEA:     {args.gmt_file if args.gmt_file else 'disabled'}")
    print(f"Enrichr:  {','.join(enrichr_libraries) if enrichr_libraries else 'disabled'}")
    print(f"Workers:  {args.workers}")
    print(f"Files:    {len(paths):,}")
    print("=" * 80)

    accumulators: Dict[str, Dict] = {}
    summaries = []
    annotation_tables = []
    annotation_fraction_reps = []
    if args.workers > 1:
        from concurrent.futures import ProcessPoolExecutor

        tasks = [(str(path), args) for path in paths]
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            for i, (annot, frac_rep, summary, local_acc) in enumerate(
                executor.map(process_one_confidence_csv_worker, tasks, chunksize=args.worker_chunksize),
                start=1,
            ):
                print(f"[{i}/{len(paths)}] done {summary.get('confidence_csv', '')}")
                summaries.append(summary)
                merge_accumulator_dict(accumulators, local_acc)
                if not annot.empty:
                    annotation_tables.append(annot)
                if not frac_rep.empty:
                    annotation_fraction_reps.append(frac_rep)
    else:
        for i, path in enumerate(paths, start=1):
            print(f"[{i}/{len(paths)}] {path}")
            annot, frac_rep, summary = process_one_confidence_csv(path, args, accumulators)
            summaries.append(summary)
            if not annot.empty:
                annotation_tables.append(annot)
            if not frac_rep.empty:
                annotation_fraction_reps.append(frac_rep)

    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(out_dir / "pair_file_summary.csv", index=False)

    if annotation_tables:
        annotation_df = pd.concat(annotation_tables, ignore_index=True)
        annotation_df.to_csv(out_dir / "cell_annotation_transitions.csv", index=False)
        frac = annotation_df.groupby(["tumor_type", "metric", "target_annotation"], as_index=False)["n_pairs"].sum()
        frac["fraction_within_tumor_metric"] = frac["n_pairs"] / frac.groupby(["tumor_type", "metric"])["n_pairs"].transform("sum")
        frac.to_csv(out_dir / "cell_annotation_fraction_by_tumor_metric.csv", index=False)
        for (tumor_type, metric), group in annotation_df.groupby(["tumor_type", "metric"], sort=False):
            safe_tumor = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(tumor_type))[:80]
            plot_annotation_fractions(
                group,
                plots_dir / f"target_annotation_fraction_{safe_tumor}_{metric}.png",
                f"Target annotation fractions: {tumor_type} / {metric}",
                args.top_annotations_plot,
            )
            if args.transition_heatmaps:
                plot_transition_heatmap(
                    group,
                    plots_dir / f"annotation_transition_heatmap_{safe_tumor}_{metric}.png",
                    f"High-confidence pre-to-post annotation transitions: {tumor_type} / {metric}",
                    args.top_annotations_plot,
                )
        transition_enrich = transition_enrichment_tests(annotation_df)
        if not transition_enrich.empty:
            transition_enrich.to_csv(out_dir / "cell_annotation_transition_enrichment_fisher.csv", index=False)

    if annotation_fraction_reps:
        frac_rep_df = pd.concat(annotation_fraction_reps, ignore_index=True)
        frac_rep_df.to_csv(out_dir / "cell_annotation_fraction_replicates.csv", index=False)
        frac_test = paired_cell_fraction_tests(frac_rep_df)
        if not frac_test.empty:
            frac_test.to_csv(out_dir / "cell_annotation_fraction_paired_tests.csv", index=False)
            for (tumor_type, metric), group in frac_test.groupby(["tumor_type", "metric"], sort=False):
                safe_tumor = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(tumor_type))[:80]
                plot_cell_fraction_delta_tests(
                    group,
                    plots_dir / f"cell_fraction_delta_tests_{safe_tumor}_{metric}.png",
                    f"Paired cell fraction shifts: {tumor_type} / {metric}",
                    args.top_annotations_plot,
                )

    tumor_summary_rows = []
    de_results: Dict[Tuple[str, str, str], pd.DataFrame] = {}
    signature_rows = []
    for _key, acc in accumulators.items():
        gene_df = paired_pseudobulk_de(acc)
        scope = acc.get("scope", "tumor_type")
        metric = acc.get("metric", "unknown_metric")
        tumor_type = acc.get("tumor_type", "unknown")
        cell_type = acc.get("cell_type", "")
        safe_tumor = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(tumor_type))[:80]
        safe_cell = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(cell_type))[:80] if cell_type else ""
        label_part = f"{safe_tumor}_{safe_cell}" if safe_cell else safe_tumor
        prefix = f"formal_pseudobulk_de_{scope}_{label_part}_{metric}"
        gene_csv = out_dir / f"{prefix}.csv"
        top_csv = out_dir / f"top_de_genes_{scope}_{safe_tumor}_{metric}.csv"
        gene_df.to_csv(gene_csv, index=False)
        gene_df.head(200).to_csv(top_csv, index=False)
        de_results[(scope, tumor_type, metric)] = gene_df
        plot_top_genes(
            gene_df,
            plots_dir / f"top_gene_deltas_{scope}_{safe_tumor}_{metric}.png",
            f"Top matched pre/post gene deltas: {tumor_type} / {metric}",
            args.top_genes_plot,
        )
        plot_volcano(
            gene_df,
            plots_dir / f"volcano_{scope}_{safe_tumor}_{metric}.png",
            f"Paired pseudobulk DE: {tumor_type} / {metric}",
            args.fdr_cutoff,
        )
        plot_replicate_delta_heatmap(
            acc,
            gene_df,
            plots_dir / f"replicate_delta_heatmap_{scope}_{safe_tumor}_{metric}.png",
            f"Replicate post-pre deltas: {tumor_type} / {metric}",
            min(args.top_genes_plot, 40),
        )
        if gene_sets is not None:
            gsea_df = preranked_gsea(gene_df, gene_sets, args.gsea_permutations, args.gsea_seed, args.gsea_min_overlap)
            gsea_csv = out_dir / f"gsea_preranked_{scope}_{safe_tumor}_{metric}.csv"
            gsea_df.to_csv(gsea_csv, index=False)
            plot_gsea_bar(
                gsea_df,
                plots_dir / f"gsea_preranked_{scope}_{safe_tumor}_{metric}.png",
                f"Preranked GSEA: {tumor_type} / {metric}",
                top_n=20,
            )
        if args.module_score_analysis:
            sig_df = paired_signature_tests(acc, DEFAULT_SIGNATURES)
            if not sig_df.empty:
                sig_csv = out_dir / f"module_signature_tests_{scope}_{label_part}_{metric}.csv"
                sig_df.to_csv(sig_csv, index=False)
                signature_rows.append(sig_df)
                plot_signature_deltas(
                    sig_df,
                    plots_dir / f"module_signature_deltas_{scope}_{label_part}_{metric}.png",
                    f"Treatment-response signature shifts: {tumor_type} / {metric}",
                )
        enrichr_outputs = []
        if enrichr_libraries:
            enrichr_outputs = run_enrichr_public(
                gene_df,
                enrichr_libraries,
                args.fdr_cutoff,
                out_dir / f"public_enrichment_{scope}_{safe_tumor}_{metric}",
            )
        tumor_summary_rows.append(
            {
                "scope": scope,
                "tumor_type": tumor_type,
                "cell_type": cell_type,
                "metric": metric,
                "n_pairs": int(acc["n_pairs"]),
                "n_replicates": len(acc["replicate_ids"]),
                "n_fdr_0p05": int((gene_df["paired_t_fdr_bh"] <= 0.05).sum()),
                "n_fdr_0p10": int((gene_df["paired_t_fdr_bh"] <= 0.10).sum()),
                "gene_csv": str(gene_csv),
                "top_gene_csv": str(top_csv),
                "enrichr_outputs": ";".join(str(x) for x in enrichr_outputs),
            }
        )
    pd.DataFrame(tumor_summary_rows).to_csv(out_dir / "formal_de_outputs_by_tumor_metric.csv", index=False)
    if signature_rows:
        pd.concat(signature_rows, ignore_index=True).to_csv(out_dir / "module_signature_tests_all.csv", index=False)
    if args.robustness_report:
        robust = confidence_metric_robustness(de_results, args.fdr_cutoff, args.robustness_top_n)
        if not robust.empty:
            robust.to_csv(out_dir / "confidence_metric_robustness_report.csv", index=False)

    print("=" * 80)
    print("Done")
    print("=" * 80)
    print(f"Summary: {out_dir / 'pair_file_summary.csv'}")
    print(f"Plots:   {plots_dir}")


if __name__ == "__main__":
    main()
