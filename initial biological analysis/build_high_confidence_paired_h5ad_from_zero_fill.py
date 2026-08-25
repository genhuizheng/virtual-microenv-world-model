#!/usr/bin/env python3
"""
Build high-confidence paired h5ad training shards from the full zero-filled data.

This script starts from the original zero-filled expression tree:

    batch_processed_unified_genes_zero_fill/
      <patient_id>/<dataset_id>/
        baseline.h5ad
        posttreatment*.h5ad
        nonzero_pairs_source_to_target/
          nonzero_pairs_source_to_target_baseline_to_posttreatment*.csv

It does NOT start from a random 50k/500k paired sample. It filters all available
MOSCOT pair CSV rows locally, then materializes paired h5ad shards in the same
format used by sample_paired_h5ad_dataloader.py.

Default filtering is local per pair file, i.e. per patient/dataset/transition.
This avoids using one global threshold across biologically different datasets.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import argparse
import gc
import json
import re
import shutil

import numpy as np
import pandas as pd

ad = None
sp = None


DEFAULT_BASE_DIR = Path("batch_processed_unified_genes_zero_fill")
DEFAULT_OUT_DIR = Path("paired_training_h5ad_high_conf_all_local")

PAIR_SUBDIR = "nonzero_pairs_source_to_target"
PAIR_PREFIX = "nonzero_pairs_source_to_target_"
PAIR_SUFFIX = ".csv"
PAIR_DIRECTION_CONFIG = {
    "source_to_target": ("nonzero_pairs_source_to_target", "nonzero_pairs_source_to_target_"),
    "target_to_source": ("nonzero_pairs_target_to_source", "nonzero_pairs_target_to_source_"),
}
EPS = 1e-8
DEFAULT_EDGE_MIN_PROBABILITY = 1e-4

CONFIDENCE_METRICS = [
    "probability",
    "row_probability",
    "max_coupling",
    "entropy_confidence",
    "top12_margin",
    "top12_ratio_log",
    "effective_target_confidence",
    "retained_mass",
    "combined_concentration",
]


def require_h5ad_stack() -> None:
    """Import h5ad dependencies only when h5ad files are inspected or written."""
    global ad, sp
    if ad is None:
        import anndata as _ad

        ad = _ad
    if sp is None:
        import scipy.sparse as _sp

        sp = _sp


def clean_str(x) -> str:
    if pd.isna(x):
        return ""
    s = str(x).strip()
    if s.lower() in {"", "nan", "none", "null", "na"}:
        return ""
    return s


def strip_moscot_concat_suffix(cell_id) -> str:
    return re.sub(r"_[01]$", "", clean_str(cell_id))


def parse_transition_from_pair_file(pair_path: Path, pair_direction: str = "source_to_target") -> str:
    name = pair_path.name
    _subdir, prefix = PAIR_DIRECTION_CONFIG.get(pair_direction, PAIR_DIRECTION_CONFIG["source_to_target"])
    if name.startswith(prefix) and name.endswith(PAIR_SUFFIX):
        return name[len(prefix):-len(PAIR_SUFFIX)]
    return pair_path.stem


def infer_target_h5ad_from_transition(transition: str) -> str:
    transition = clean_str(transition)
    if not transition.startswith("baseline_to_"):
        return ""
    target_label = transition.replace("baseline_to_", "", 1)
    if target_label == "posttreatment":
        return "posttreatment.h5ad"
    return f"{target_label}.h5ad"


def count_rows_fast(csv_path: Path) -> int:
    with open(csv_path, "rb") as f:
        return max(sum(1 for _ in f) - 1, 0)


def to_csr_matrix(x):
    require_h5ad_stack()
    if sp.issparse(x):
        return x.tocsr()
    return sp.csr_matrix(x)


def safe_obs_value_frame(obs: pd.DataFrame) -> pd.DataFrame:
    out = obs.copy()
    for col in out.columns:
        if pd.api.types.is_object_dtype(out[col]) or isinstance(out[col].dtype, pd.CategoricalDtype):
            out[col] = out[col].astype(str)
    return out


def discover_pair_files(base_dir: Path, pair_direction: str = "source_to_target") -> List[Path]:
    directions = ["source_to_target", "target_to_source"] if pair_direction == "both" else [pair_direction]
    pair_files = []
    for direction in directions:
        subdir, prefix = PAIR_DIRECTION_CONFIG[direction]
        pattern = f"{subdir}/{prefix}*{PAIR_SUFFIX}"
        pair_files.extend(sorted(base_dir.rglob(pattern)))
    if not pair_files:
        raise FileNotFoundError(f"No pair files found under {base_dir} for pair_direction={pair_direction}")
    return pair_files


def infer_pair_file_metadata(pair_path: Path, base_dir: Path) -> Dict:
    rel = pair_path.relative_to(base_dir)
    parts = rel.parts
    patient_id = parts[0] if len(parts) >= 1 else ""
    dataset_id = parts[1] if len(parts) >= 2 else ""
    pair_subdir = pair_path.parent.name
    pair_direction = "target_to_source" if pair_subdir == "nonzero_pairs_target_to_source" else "source_to_target"
    transition = parse_transition_from_pair_file(pair_path, pair_direction)
    dataset_dir = base_dir / patient_id / dataset_id
    baseline_h5ad = base_dir / patient_id / dataset_id / "baseline.h5ad"
    target_h5ad = base_dir / patient_id / dataset_id / infer_target_h5ad_from_transition(transition)
    return {
        "pair_file": str(pair_path),
        "relative_pair_file": rel.as_posix(),
        "patient_id": patient_id,
        "dataset_id": dataset_id,
        "dataset_dir": str(dataset_dir),
        "pair_direction": pair_direction,
        "transition": transition,
        "baseline_h5ad": str(baseline_h5ad),
        "target_h5ad": str(target_h5ad),
        "relative_baseline_h5ad": baseline_h5ad.relative_to(base_dir).as_posix() if baseline_h5ad.exists() else "",
        "relative_target_h5ad": target_h5ad.relative_to(base_dir).as_posix() if target_h5ad.exists() else "",
        "baseline_h5ad_exists": int(baseline_h5ad.exists()),
        "target_h5ad_exists": int(target_h5ad.exists()),
    }


def validate_pair_matches_h5ad(pair_file: Path, baseline_h5ad: Path, target_h5ad: Path, nrows_check: int) -> Dict:
    out = {
        "match_check_status": "not_checked",
        "source_match_fraction": 0.0,
        "target_match_fraction": 0.0,
        "n_check_rows": 0,
    }
    if not pair_file.exists():
        out["match_check_status"] = "missing_pair_file"
        return out
    if not baseline_h5ad.exists():
        out["match_check_status"] = "missing_baseline_h5ad"
        return out
    if not target_h5ad.exists():
        out["match_check_status"] = "missing_target_h5ad"
        return out

    try:
        require_h5ad_stack()
        df = pd.read_csv(pair_file, nrows=nrows_check)
        required = {"source_cell", "target_cell", "probability"}
        missing = required - set(df.columns)
        if missing:
            out["match_check_status"] = "missing_required_columns:" + ",".join(sorted(missing))
            return out

        src = set(df["source_cell"].map(strip_moscot_concat_suffix))
        tgt = set(df["target_cell"].map(strip_moscot_concat_suffix))
        a = ad.read_h5ad(baseline_h5ad, backed="r")
        b = ad.read_h5ad(target_h5ad, backed="r")
        src_obs = set(a.obs_names)
        tgt_obs = set(b.obs_names)
        a.file.close()
        b.file.close()

        out.update(
            {
                "match_check_status": "ok",
                "n_check_rows": int(len(df)),
                "source_match_fraction": float(len(src & src_obs) / len(src)) if src else 0.0,
                "target_match_fraction": float(len(tgt & tgt_obs) / len(tgt)) if tgt else 0.0,
            }
        )
        return out
    except Exception as exc:
        out["match_check_status"] = "error:" + repr(exc)
        return out


def scan_pair_files(base_dir: Path, do_match_check: bool, match_check_rows: int, pair_direction: str = "source_to_target") -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    pair_files = discover_pair_files(base_dir, pair_direction=pair_direction)
    for i, pair_file in enumerate(pair_files, start=1):
        meta = infer_pair_file_metadata(pair_file, base_dir)
        columns = list(pd.read_csv(pair_file, nrows=1).columns)
        row = dict(meta)
        row["n_pairs_available"] = count_rows_fast(pair_file)
        row["columns"] = ",".join(columns)
        row["has_required_columns"] = int({"source_cell", "target_cell", "probability"}.issubset(columns))
        if do_match_check and row["n_pairs_available"] > 0:
            row.update(
                validate_pair_matches_h5ad(
                    pair_file,
                    Path(row["baseline_h5ad"]),
                    Path(row["target_h5ad"]),
                    match_check_rows,
                )
            )
        rows.append(row)
        print(
            f"[scan {i}/{len(pair_files)}] {row['relative_pair_file']} "
            f"pairs={row['n_pairs_available']:,} required={row['has_required_columns']} "
            f"match={row.get('source_match_fraction', 'NA')}/{row.get('target_match_fraction', 'NA')}"
        )

    all_df = pd.DataFrame(rows).fillna("")
    valid = all_df[
        (pd.to_numeric(all_df["n_pairs_available"], errors="coerce").fillna(0).astype(int) > 0)
        & (all_df["has_required_columns"].astype(int) == 1)
        & (all_df["baseline_h5ad_exists"].astype(int) == 1)
        & (all_df["target_h5ad_exists"].astype(int) == 1)
    ].copy()
    if do_match_check:
        valid = valid[
            (valid["match_check_status"] == "ok")
            & (pd.to_numeric(valid["source_match_fraction"], errors="coerce").fillna(0) > 0.99)
            & (pd.to_numeric(valid["target_match_fraction"], errors="coerce").fillna(0) > 0.99)
        ].copy()
    if valid.empty:
        raise ValueError("No valid pair files remain after scanning.")
    return all_df, valid


def numeric_col(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        raise ValueError(f"Missing confidence column '{col}'. Existing columns: {list(df.columns)}")
    return pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], np.nan)


def read_h5ad_n_obs(path: Path) -> int:
    require_h5ad_stack()
    adata = ad.read_h5ad(path, backed="r")
    try:
        return int(adata.n_obs)
    finally:
        adata.file.close()


def percentile_rank(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    if numeric.notna().sum() <= 1:
        return pd.Series(np.ones(len(values)), index=values.index, dtype=float)
    return numeric.rank(method="average", pct=True).fillna(0.0)


def add_local_ot_confidence_metrics(df: pd.DataFrame, source_n_obs: int, target_n_obs: int) -> pd.DataFrame:
    """Compute confidence metrics from the full local nonzero OT table."""
    out = df.copy()
    out["source_cell_moscot"] = out["source_cell"].astype(str)
    out["target_cell_moscot"] = out["target_cell"].astype(str)
    out["source_cell_original_obs"] = out["source_cell_moscot"].map(strip_moscot_concat_suffix)
    out["target_cell_original_obs"] = out["target_cell_moscot"].map(strip_moscot_concat_suffix)
    out["probability"] = pd.to_numeric(out["probability"], errors="coerce").fillna(0.0).clip(lower=0.0)

    source_key = "source_cell_original_obs"
    target_key = "target_cell_original_obs"
    row_sum = out.groupby(source_key, sort=False)["probability"].transform("sum")
    out["ot_row_sum"] = row_sum
    out["ot_p_row_norm"] = out["probability"] / (row_sum + EPS)
    col_sum = out.groupby(target_key, sort=False)["probability"].transform("sum")
    out["ot_col_sum"] = col_sum
    out["ot_p_col_norm"] = out["probability"] / (col_sum + EPS)

    ordered = out.sort_values([source_key, "ot_p_row_norm"], ascending=[True, False]).copy()
    ordered["ot_target_rank_in_source"] = ordered.groupby(source_key, sort=False).cumcount() + 1
    rank_map = ordered["ot_target_rank_in_source"]
    out["ot_target_rank_in_source"] = rank_map.reindex(out.index).astype(int)
    out["ot_is_top1_target"] = (out["ot_target_rank_in_source"] == 1).astype(int)

    grouped_p = ordered.groupby(source_key, sort=False)["ot_p_row_norm"]
    top1 = grouped_p.first().rename("source_max_coupling")
    top2 = grouped_p.nth(1).rename("source_second_coupling")
    out["source_max_coupling"] = out[source_key].map(top1).fillna(0.0)
    out["source_second_coupling"] = out[source_key].map(top2).fillna(0.0)
    out["source_top12_margin"] = out["source_max_coupling"] - out["source_second_coupling"]
    out["source_top12_ratio"] = out["source_max_coupling"] / (out["source_second_coupling"] + EPS)
    out["source_top12_ratio_log"] = np.log(out["source_top12_ratio"] + EPS)

    entropy_terms = -(out["ot_p_row_norm"] * np.log(out["ot_p_row_norm"] + EPS))
    entropy = entropy_terms.groupby(out[source_key], sort=False).sum()
    n_target = max(int(target_n_obs), 2)
    entropy_conf = (1.0 - entropy / np.log(n_target)).clip(lower=0.0, upper=1.0)
    n_eff = np.exp(entropy)
    effective_conf = (1.0 - ((n_eff - 1.0) / max(n_target - 1.0, 1.0))).clip(lower=0.0, upper=1.0)

    out["source_entropy"] = out[source_key].map(entropy).fillna(0.0)
    out["source_entropy_confidence"] = out[source_key].map(entropy_conf).fillna(0.0)
    out["source_effective_n_targets"] = out[source_key].map(n_eff).fillna(0.0)
    out["source_effective_target_fraction"] = out["source_effective_n_targets"] / float(n_target)
    out["source_effective_target_confidence"] = out[source_key].map(effective_conf).fillna(0.0)
    out["source_retained_mass"] = out["ot_row_sum"]

    target_ordered = out.sort_values([target_key, "ot_p_col_norm"], ascending=[True, False]).copy()
    target_ordered["ot_source_rank_in_target"] = target_ordered.groupby(target_key, sort=False).cumcount() + 1
    out["ot_source_rank_in_target"] = target_ordered["ot_source_rank_in_target"].reindex(out.index).astype(int)
    out["ot_is_top1_source"] = (out["ot_source_rank_in_target"] == 1).astype(int)
    out["ot_is_mutual_top1"] = ((out["ot_is_top1_target"] == 1) & (out["ot_is_top1_source"] == 1)).astype(int)

    grouped_col_p = target_ordered.groupby(target_key, sort=False)["ot_p_col_norm"]
    target_top1 = grouped_col_p.first().rename("target_max_coupling")
    target_top2 = grouped_col_p.nth(1).rename("target_second_coupling")
    out["target_max_coupling"] = out[target_key].map(target_top1).fillna(0.0)
    out["target_second_coupling"] = out[target_key].map(target_top2).fillna(0.0)
    out["target_top12_margin"] = out["target_max_coupling"] - out["target_second_coupling"]
    out["target_top12_ratio"] = out["target_max_coupling"] / (out["target_second_coupling"] + EPS)
    out["target_top12_ratio_log"] = np.log(out["target_top12_ratio"] + EPS)

    target_entropy_terms = -(out["ot_p_col_norm"] * np.log(out["ot_p_col_norm"] + EPS))
    target_entropy = target_entropy_terms.groupby(out[target_key], sort=False).sum()
    n_source = max(int(source_n_obs), 2)
    target_entropy_conf = (1.0 - target_entropy / np.log(n_source)).clip(lower=0.0, upper=1.0)
    target_n_eff = np.exp(target_entropy)
    target_effective_conf = (1.0 - ((target_n_eff - 1.0) / max(n_source - 1.0, 1.0))).clip(lower=0.0, upper=1.0)

    out["target_entropy"] = out[target_key].map(target_entropy).fillna(0.0)
    out["target_entropy_confidence"] = out[target_key].map(target_entropy_conf).fillna(0.0)
    out["target_effective_n_sources"] = out[target_key].map(target_n_eff).fillna(0.0)
    out["target_effective_source_fraction"] = out["target_effective_n_sources"] / float(n_source)
    out["target_effective_source_confidence"] = out[target_key].map(target_effective_conf).fillna(0.0)
    out["target_retained_mass"] = out["ot_col_sum"]

    source_metrics = out[[source_key, "source_max_coupling", "source_entropy_confidence", "source_top12_margin"]].drop_duplicates(source_key)
    source_metrics["source_max_coupling_rank"] = percentile_rank(source_metrics["source_max_coupling"])
    source_metrics["source_entropy_confidence_rank"] = percentile_rank(source_metrics["source_entropy_confidence"])
    source_metrics["source_top12_margin_rank"] = percentile_rank(source_metrics["source_top12_margin"])
    source_metrics["source_combined_concentration"] = (
        0.40 * source_metrics["source_entropy_confidence_rank"]
        + 0.30 * source_metrics["source_top12_margin_rank"]
        + 0.30 * source_metrics["source_max_coupling_rank"]
    )
    rank_cols = [
        "source_max_coupling_rank",
        "source_entropy_confidence_rank",
        "source_top12_margin_rank",
        "source_combined_concentration",
    ]
    out = out.merge(source_metrics[[source_key] + rank_cols], on=source_key, how="left")

    target_metrics = out[[target_key, "target_max_coupling", "target_entropy_confidence", "target_top12_margin"]].drop_duplicates(target_key)
    target_metrics["target_max_coupling_rank"] = percentile_rank(target_metrics["target_max_coupling"])
    target_metrics["target_entropy_confidence_rank"] = percentile_rank(target_metrics["target_entropy_confidence"])
    target_metrics["target_top12_margin_rank"] = percentile_rank(target_metrics["target_top12_margin"])
    target_metrics["target_combined_concentration"] = (
        0.40 * target_metrics["target_entropy_confidence_rank"]
        + 0.30 * target_metrics["target_top12_margin_rank"]
        + 0.30 * target_metrics["target_max_coupling_rank"]
    )
    target_rank_cols = [
        "target_max_coupling_rank",
        "target_entropy_confidence_rank",
        "target_top12_margin_rank",
        "target_combined_concentration",
    ]
    out = out.merge(target_metrics[[target_key] + target_rank_cols], on=target_key, how="left")
    out["both_combined_concentration"] = 0.5 * (
        out["source_combined_concentration"] + out["target_combined_concentration"]
    )
    out["local_source_n_obs"] = n_source
    out["local_target_n_obs"] = n_target
    return out


def metric_to_score(df: pd.DataFrame, metric: str, axis: str = "source") -> pd.Series:
    if metric == "probability":
        return df["probability"]
    if metric == "row_probability" and axis == "target":
        return df["ot_p_col_norm"]
    if metric == "row_probability" and axis == "both":
        return 0.5 * (df["ot_p_row_norm"] + df["ot_p_col_norm"])
    if metric == "row_probability":
        return df["ot_p_row_norm"]
    prefix = "target" if axis == "target" else "source"
    if axis == "both" and metric == "combined_concentration":
        return df["both_combined_concentration"]
    if metric == "max_coupling":
        if axis == "both":
            return 0.5 * (df["source_max_coupling"] + df["target_max_coupling"])
        return df[f"{prefix}_max_coupling"]
    if metric == "entropy_confidence":
        if axis == "both":
            return 0.5 * (df["source_entropy_confidence"] + df["target_entropy_confidence"])
        return df[f"{prefix}_entropy_confidence"]
    if metric == "top12_margin":
        if axis == "both":
            return 0.5 * (df["source_top12_margin"] + df["target_top12_margin"])
        return df[f"{prefix}_top12_margin"]
    if metric == "top12_ratio_log":
        if axis == "both":
            return 0.5 * (df["source_top12_ratio_log"] + df["target_top12_ratio_log"])
        return df[f"{prefix}_top12_ratio_log"]
    if metric == "effective_target_confidence":
        if axis == "target":
            return df["target_effective_source_confidence"]
        if axis == "both":
            return 0.5 * (df["source_effective_target_confidence"] + df["target_effective_source_confidence"])
        return df["source_effective_target_confidence"]
    if metric == "retained_mass":
        if axis == "both":
            return 0.5 * (df["source_retained_mass"] + df["target_retained_mass"])
        return df[f"{prefix}_retained_mass"]
    if metric == "combined_concentration":
        return df[f"{prefix}_combined_concentration"]
    raise ValueError(f"Unknown confidence metric: {metric}")


def filter_one_pair_file(row: pd.Series, args: argparse.Namespace, running_start_index: int) -> Tuple[pd.DataFrame, Dict]:
    pair_file = Path(row["pair_file"])
    df = pd.read_csv(pair_file)
    required = {"source_cell", "target_cell", "probability"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{pair_file} missing required columns: {missing}")

    source_n_obs = read_h5ad_n_obs(Path(row["baseline_h5ad"]))
    target_n_obs = read_h5ad_n_obs(Path(row["target_h5ad"]))
    df = add_local_ot_confidence_metrics(df, source_n_obs=source_n_obs, target_n_obs=target_n_obs)
    if args.pair_selection in {"top1", "top1_source"}:
        candidate = df[df["ot_is_top1_target"] == 1].copy()
    elif args.pair_selection == "top1_target":
        candidate = df[df["ot_is_top1_source"] == 1].copy()
    elif args.pair_selection == "mutual_top1":
        candidate = df[df["ot_is_mutual_top1"] == 1].copy()
    elif args.pair_selection == "all":
        candidate = df.copy()
    else:
        raise ValueError(f"Unknown pair selection: {args.pair_selection}")

    confidence = metric_to_score(candidate, args.confidence_metric, args.confidence_axis)
    finite = confidence[np.isfinite(confidence.to_numpy())]
    if finite.empty:
        threshold = np.inf
    elif args.top_fraction is not None:
        threshold = float(np.quantile(finite, 1.0 - args.top_fraction))
        if args.min_confidence is not None:
            threshold = max(threshold, float(args.min_confidence))
    elif args.min_confidence is not None:
        threshold = float(args.min_confidence)
    else:
        raise ValueError("Provide --top-fraction, --min-confidence, or both.")

    keep = confidence >= threshold
    keep = keep.fillna(False)
    filtered = candidate.loc[keep].copy()

    filtered["patient_id"] = row["patient_id"]
    filtered["dataset_id"] = row["dataset_id"]
    filtered["transition"] = row["transition"]
    filtered["source_pair_file"] = row["relative_pair_file"]
    filtered["baseline_h5ad"] = row["relative_baseline_h5ad"]
    filtered["target_h5ad"] = row["relative_target_h5ad"]
    filtered["confidence_score"] = metric_to_score(filtered, args.confidence_metric, args.confidence_axis)
    filtered["confidence_filter_metric"] = args.confidence_metric
    filtered["confidence_filter_axis"] = args.confidence_axis
    filtered["confidence_filter_threshold"] = threshold
    filtered["confidence_filter_scope"] = "pair_file"
    filtered["confidence_pair_selection"] = args.pair_selection

    filtered["pair_id"] = [
        f"{row['patient_id']}|{row['dataset_id']}|{row['transition']}|highconf_pair_{running_start_index + j}"
        for j in range(len(filtered))
    ]

    first_cols = [
        "pair_id",
        "patient_id",
        "dataset_id",
        "transition",
        "source_time",
        "target_time",
        "source_cell_moscot",
        "target_cell_moscot",
        "source_cell_original_obs",
        "target_cell_original_obs",
        "probability",
        "ot_p_row_norm",
        "confidence_score",
        "baseline_h5ad",
        "target_h5ad",
        "source_pair_file",
        "confidence_filter_metric",
        "confidence_filter_axis",
        "confidence_filter_threshold",
        "confidence_filter_scope",
        "confidence_pair_selection",
        "ot_target_rank_in_source",
        "ot_is_top1_target",
        "ot_source_rank_in_target",
        "ot_is_top1_source",
        "ot_is_mutual_top1",
        "source_max_coupling",
        "source_second_coupling",
        "source_top12_margin",
        "source_top12_ratio",
        "source_top12_ratio_log",
        "source_entropy_confidence",
        "source_effective_n_targets",
        "source_effective_target_fraction",
        "source_effective_target_confidence",
        "source_retained_mass",
        "source_combined_concentration",
        "target_max_coupling",
        "target_second_coupling",
        "target_top12_margin",
        "target_top12_ratio",
        "target_top12_ratio_log",
        "target_entropy_confidence",
        "target_effective_n_sources",
        "target_effective_source_fraction",
        "target_effective_source_confidence",
        "target_retained_mass",
        "target_combined_concentration",
        "both_combined_concentration",
        "local_source_n_obs",
        "local_target_n_obs",
    ]
    other_cols = [c for c in filtered.columns if c not in first_cols and c not in {"source_cell", "target_cell"}]
    filtered = filtered[first_cols + other_cols]

    summary = {
        "relative_pair_file": row["relative_pair_file"],
        "patient_id": row["patient_id"],
        "dataset_id": row["dataset_id"],
        "transition": row["transition"],
        "confidence_metric": args.confidence_metric,
        "confidence_axis": args.confidence_axis,
        "pair_selection": args.pair_selection,
        "threshold_scope": "pair_file",
        "threshold": threshold,
        "n_pairs_before": int(len(df)),
        "n_candidate_pairs": int(len(candidate)),
        "n_pairs_after": int(len(filtered)),
        "retained_fraction_of_all_nonzero_pairs": float(len(filtered) / len(df)) if len(df) else 0.0,
        "retained_fraction_of_candidates": float(len(filtered) / len(candidate)) if len(candidate) else 0.0,
        "confidence_min": float(np.nanmin(confidence)) if len(df) else np.nan,
        "confidence_median": float(np.nanmedian(confidence)) if len(df) else np.nan,
        "confidence_max": float(np.nanmax(confidence)) if len(df) else np.nan,
        "source_n_obs": int(source_n_obs),
        "target_n_obs": int(target_n_obs),
        "unique_sources": int(df["source_cell_original_obs"].nunique()),
        "unique_targets_in_nonzero_pairs": int(df["target_cell_original_obs"].nunique()),
        "source_max_coupling_median": float(df[["source_cell_original_obs", "source_max_coupling"]].drop_duplicates()["source_max_coupling"].median()),
        "source_entropy_confidence_median": float(df[["source_cell_original_obs", "source_entropy_confidence"]].drop_duplicates()["source_entropy_confidence"].median()),
        "source_top12_margin_median": float(df[["source_cell_original_obs", "source_top12_margin"]].drop_duplicates()["source_top12_margin"].median()),
        "target_max_coupling_median": float(df[["target_cell_original_obs", "target_max_coupling"]].drop_duplicates()["target_max_coupling"].median()),
        "target_entropy_confidence_median": float(df[["target_cell_original_obs", "target_entropy_confidence"]].drop_duplicates()["target_entropy_confidence"].median()),
        "target_top12_margin_median": float(df[["target_cell_original_obs", "target_top12_margin"]].drop_duplicates()["target_top12_margin"].median()),
    }
    return filtered, summary


def select_candidate_edges(df: pd.DataFrame, pair_selection: str) -> pd.DataFrame:
    if pair_selection in {"top1", "top1_source"}:
        return df[df["ot_is_top1_target"] == 1].copy()
    if pair_selection == "top1_target":
        return df[df["ot_is_top1_source"] == 1].copy()
    if pair_selection == "mutual_top1":
        return df[df["ot_is_mutual_top1"] == 1].copy()
    if pair_selection == "all":
        return df.copy()
    raise ValueError(f"Unknown pair selection: {pair_selection}")


def prepare_one_pair_file_for_analysis(row: pd.Series, args: argparse.Namespace, confidence_metric: Optional[str] = None) -> Tuple[pd.DataFrame, Dict]:
    pair_file = Path(row["pair_file"])
    df = pd.read_csv(pair_file)
    required = {"source_cell", "target_cell", "probability"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{pair_file} missing required columns: {missing}")

    source_n_obs = read_h5ad_n_obs(Path(row["baseline_h5ad"]))
    target_n_obs = read_h5ad_n_obs(Path(row["target_h5ad"]))
    df = add_local_ot_confidence_metrics(df, source_n_obs=source_n_obs, target_n_obs=target_n_obs)
    candidate = select_candidate_edges(df, args.pair_selection)
    metric = confidence_metric or args.confidence_metric
    confidence = metric_to_score(candidate, metric, args.confidence_axis)

    candidate["patient_id"] = row["patient_id"]
    candidate["dataset_id"] = row["dataset_id"]
    candidate["transition"] = row["transition"]
    candidate["pair_direction"] = row.get("pair_direction", args.pair_direction)
    candidate["source_pair_file"] = row["relative_pair_file"]
    candidate["baseline_h5ad"] = row["relative_baseline_h5ad"]
    candidate["target_h5ad"] = row["relative_target_h5ad"]
    candidate["confidence_score"] = confidence
    candidate["confidence_metric"] = metric
    candidate["confidence_axis"] = args.confidence_axis
    candidate["confidence_pair_selection"] = args.pair_selection

    candidate["edge_id"] = [
        f"{row['patient_id']}|{row['dataset_id']}|{row['transition']}|edge_{j}"
        for j in range(len(candidate))
    ]

    first_cols = [
        "edge_id",
        "patient_id",
        "dataset_id",
        "transition",
        "pair_direction",
        "source_time",
        "target_time",
        "source_cell_moscot",
        "target_cell_moscot",
        "source_cell_original_obs",
        "target_cell_original_obs",
        "probability",
        "ot_p_row_norm",
        "ot_p_col_norm",
        "confidence_score",
        "confidence_metric",
        "confidence_axis",
        "confidence_pair_selection",
        "baseline_h5ad",
        "target_h5ad",
        "source_pair_file",
        "ot_target_rank_in_source",
        "ot_is_top1_target",
        "ot_source_rank_in_target",
        "ot_is_top1_source",
        "ot_is_mutual_top1",
        "source_max_coupling",
        "source_second_coupling",
        "source_top12_margin",
        "source_top12_ratio",
        "source_top12_ratio_log",
        "source_entropy_confidence",
        "source_effective_n_targets",
        "source_effective_target_fraction",
        "source_effective_target_confidence",
        "source_retained_mass",
        "source_combined_concentration",
        "target_max_coupling",
        "target_second_coupling",
        "target_top12_margin",
        "target_top12_ratio",
        "target_top12_ratio_log",
        "target_entropy_confidence",
        "target_effective_n_sources",
        "target_effective_source_fraction",
        "target_effective_source_confidence",
        "target_retained_mass",
        "target_combined_concentration",
        "both_combined_concentration",
        "local_source_n_obs",
        "local_target_n_obs",
    ]
    other_cols = [c for c in candidate.columns if c not in first_cols and c not in {"source_cell", "target_cell"}]
    candidate = candidate[first_cols + other_cols]

    source_unique = df[["source_cell_original_obs", "source_max_coupling", "source_entropy_confidence", "source_top12_margin"]].drop_duplicates("source_cell_original_obs")
    target_unique = df[["target_cell_original_obs", "target_max_coupling", "target_entropy_confidence", "target_top12_margin"]].drop_duplicates("target_cell_original_obs")
    summary = {
        "relative_pair_file": row["relative_pair_file"],
        "patient_id": row["patient_id"],
        "dataset_id": row["dataset_id"],
        "transition": row["transition"],
        "pair_direction": row.get("pair_direction", args.pair_direction),
        "confidence_metric": metric,
        "confidence_axis": args.confidence_axis,
        "pair_selection": args.pair_selection,
        "n_nonzero_edges": int(len(df)),
        "n_output_edges": int(len(candidate)),
        "source_n_obs": int(source_n_obs),
        "target_n_obs": int(target_n_obs),
        "unique_sources_in_nonzero_edges": int(df["source_cell_original_obs"].nunique()),
        "unique_targets_in_nonzero_edges": int(df["target_cell_original_obs"].nunique()),
        "confidence_score_min": float(confidence.min()) if len(confidence) else np.nan,
        "confidence_score_median": float(confidence.median()) if len(confidence) else np.nan,
        "confidence_score_max": float(confidence.max()) if len(confidence) else np.nan,
        "source_max_coupling_median": float(source_unique["source_max_coupling"].median()),
        "source_entropy_confidence_median": float(source_unique["source_entropy_confidence"].median()),
        "source_top12_margin_median": float(source_unique["source_top12_margin"].median()),
        "target_max_coupling_median": float(target_unique["target_max_coupling"].median()),
        "target_entropy_confidence_median": float(target_unique["target_entropy_confidence"].median()),
        "target_top12_margin_median": float(target_unique["target_top12_margin"].median()),
    }
    return candidate, summary


def metrics_for_output(args: argparse.Namespace) -> List[str]:
    if args.split_confidence_metrics:
        if args.metrics_to_split.strip().lower() == "all":
            return list(CONFIDENCE_METRICS)
        metrics = [m.strip() for m in args.metrics_to_split.split(",") if m.strip()]
        bad = [m for m in metrics if m not in CONFIDENCE_METRICS]
        if bad:
            raise ValueError(f"Unknown metrics in --metrics-to-split: {bad}. Choices: {CONFIDENCE_METRICS}")
        return metrics
    return [args.confidence_metric]


def write_analysis_csv(scan_df: pd.DataFrame, valid_df: pd.DataFrame, args: argparse.Namespace) -> None:
    sep = "," if args.analysis_format == "csv" else "\t"
    suffix = "csv" if args.analysis_format == "csv" else "tsv"

    if args.write_to_dataset_dirs:
        summaries = []
        for i, row in enumerate(valid_df.to_dict(orient="records"), start=1):
            print("=" * 80)
            print(f"[analysis {i}/{len(valid_df)}] {row['relative_pair_file']}")
            print("=" * 80)
            for metric in metrics_for_output(args):
                edges, summary = prepare_one_pair_file_for_analysis(pd.Series(row), args, confidence_metric=metric)
                dataset_out_dir = Path(row["dataset_dir"]) / args.high_confidence_dir_name / metric
                dataset_out_dir.mkdir(parents=True, exist_ok=True)
                pair_direction = row.get("pair_direction", args.pair_direction)
                out_csv = dataset_out_dir / f"ot_confidence_{summary['transition']}_{pair_direction}_{metric}.{suffix}"
                if out_csv.exists() and not args.overwrite:
                    raise FileExistsError(f"Output CSV exists. Use --overwrite or choose another output: {out_csv}")
                edges.to_csv(out_csv, sep=sep, index=False)
                pd.DataFrame([summary]).to_csv(out_csv.with_name(out_csv.stem + "_summary.csv"), sep=sep, index=False)
                summaries.append(summary)
                print(f"{metric}: wrote edges={len(edges):,} -> {out_csv}")

        summary_all = pd.DataFrame(summaries)
        merged_summary = summary_all.merge(valid_df[["dataset_dir", "relative_pair_file"]], on="relative_pair_file", how="left")
        for (dataset_dir, metric), group in merged_summary.groupby(["dataset_dir", "confidence_metric"], sort=False):
            out_dir = Path(dataset_dir) / args.high_confidence_dir_name / metric
            group.to_csv(out_dir / f"ot_confidence_summary_all_{metric}.{suffix}", sep=sep, index=False)
        print("=" * 80)
        print("Analysis CSV done")
        print("=" * 80)
        print(f"CSV outputs were written inside each PATIENT/DATASET/{args.high_confidence_dir_name}/<metric> folder.")
        return

    ensure_clean_out_dir(args.out_dir, args.overwrite)
    scan_df.to_csv(args.out_dir / "paired_pair_file_scan_all.tsv", sep="\t", index=False)

    summaries = []
    for i, row in enumerate(valid_df.to_dict(orient="records"), start=1):
        print("=" * 80)
        print(f"[analysis {i}/{len(valid_df)}] {row['relative_pair_file']}")
        print("=" * 80)
        for metric in metrics_for_output(args):
            metric_dir = args.out_dir / metric
            metric_dir.mkdir(parents=True, exist_ok=True)
            edges_path = metric_dir / f"ot_confidence_edges_{metric}.{suffix}"
            edges, summary = prepare_one_pair_file_for_analysis(pd.Series(row), args, confidence_metric=metric)
            edges.to_csv(edges_path, sep=sep, index=False, mode="a", header=not edges_path.exists())
            summaries.append(summary)
            print(f"{metric}: wrote edges={len(edges):,} from nonzero_edges={summary['n_nonzero_edges']:,}")

    summary_df = pd.DataFrame(summaries)
    for metric, group in summary_df.groupby("confidence_metric", sort=False):
        metric_dir = args.out_dir / metric
        group.to_csv(metric_dir / f"ot_confidence_summary_by_pair_file_{metric}.{suffix}", sep=sep, index=False)
    config = vars(args).copy()
    config["base_dir"] = str(args.base_dir)
    config["out_dir"] = str(args.out_dir)
    (args.out_dir / "analysis_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print("=" * 80)
    print("Analysis CSV done")
    print("=" * 80)
    print(f"CSV outputs were written under: {args.out_dir}/<metric>/")


def discover_npz_matrix_jobs(base_dir: Path, matrix_direction: str) -> pd.DataFrame:
    directions = ["source_to_target", "target_to_source"] if matrix_direction == "both" else [matrix_direction]
    rows = []
    for dataset_dir in sorted(base_dir.glob("*/*")):
        if not dataset_dir.is_dir():
            continue
        matrix_root = dataset_dir / "transport_matrices"
        if not matrix_root.exists():
            continue
        patient_id = dataset_dir.parent.name
        dataset_id = dataset_dir.name

        for full_base in sorted(matrix_root.glob("transport_matrix_*")):
            if not full_base.is_dir() or full_base.name.startswith("transport_matrix_chunks_"):
                continue
            transition = full_base.name.replace("transport_matrix_", "", 1)
            for direction in directions:
                npz_path = full_base / direction / "transport_matrix.npz"
                if npz_path.exists():
                    rows.append(
                        {
                            "matrix_storage": "full",
                            "patient_id": patient_id,
                            "dataset_id": dataset_id,
                            "dataset_dir": str(dataset_dir),
                            "transition": transition,
                            "direction": direction,
                            "matrix_base": str(full_base),
                            "npz_file": str(npz_path),
                            "metadata_file": "",
                        }
                    )

        for chunk_base in sorted(matrix_root.glob("transport_matrix_chunks_*")):
            if not chunk_base.is_dir():
                continue
            transition = chunk_base.name.replace("transport_matrix_chunks_", "", 1)
            for direction in directions:
                meta_path = chunk_base / direction / "metadata.json"
                if meta_path.exists():
                    rows.append(
                        {
                            "matrix_storage": "chunked",
                            "patient_id": patient_id,
                            "dataset_id": dataset_id,
                            "dataset_dir": str(dataset_dir),
                            "transition": transition,
                            "direction": direction,
                            "matrix_base": str(chunk_base),
                            "npz_file": "",
                            "metadata_file": str(meta_path),
                        }
                    )
    df = pd.DataFrame(rows)
    if df.empty:
        raise FileNotFoundError(
            f"No MOSCOT matrix npz jobs found under {base_dir} for direction={matrix_direction}"
        )
    return df


def safe_npz_array(x) -> np.ndarray:
    arr = np.asarray(x)
    if arr.dtype.kind in {"U", "S", "O"}:
        return arr.astype(str)
    return arr


def load_transport_npz(npz_path: Path) -> Dict:
    with np.load(npz_path, allow_pickle=True) as data:
        return {
            "matrix": np.asarray(data["matrix"], dtype=np.float64),
            "source_cells": safe_npz_array(data["source_cells"]),
            "target_cells": safe_npz_array(data["target_cells"]),
            "direction": str(data["direction"]) if "direction" in data.files else "",
            "source_time": str(data["source_time"]) if "source_time" in data.files else "",
            "target_time": str(data["target_time"]) if "target_time" in data.files else "",
        }


def infer_job_from_npz_file(npz_path: Path, matrix_direction: str) -> Dict:
    direction = matrix_direction
    payload = load_transport_npz(npz_path)
    if direction == "both":
        direction = clean_str(payload.get("direction", "")) or "source_to_target"
    if direction not in {"source_to_target", "target_to_source"}:
        direction = clean_str(payload.get("direction", "")) or "source_to_target"

    transition = npz_path.parent.parent.name
    matrix_storage = "chunked" if npz_path.name.startswith("chunk_") else "full"
    if transition in {"source_to_target", "target_to_source"}:
        transition = npz_path.parent.parent.parent.name
    transition = transition.replace("transport_matrix_chunks_", "").replace("transport_matrix_", "")

    dataset_dir = npz_path
    for parent in npz_path.parents:
        if parent.name == "transport_matrices":
            dataset_dir = parent.parent
            break

    return {
        "matrix_storage": matrix_storage,
        "patient_id": dataset_dir.parent.name if dataset_dir.parent != dataset_dir else "",
        "dataset_id": dataset_dir.name,
        "dataset_dir": str(dataset_dir),
        "transition": transition,
        "direction": direction,
        "matrix_base": str(npz_path.parent),
        "npz_file": str(npz_path),
        "metadata_file": "",
    }


def top2_from_probs(probs: np.ndarray, axis: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if probs.shape[axis] == 0:
        n = probs.shape[1 - axis]
        return np.zeros(n), np.zeros(n), np.zeros(n, dtype=int)
    if axis == 1:
        top1_idx = np.argmax(probs, axis=1)
        top1 = probs[np.arange(probs.shape[0]), top1_idx]
        if probs.shape[1] > 1:
            part = np.partition(probs, -2, axis=1)
            top2 = part[:, -2]
        else:
            top2 = np.zeros(probs.shape[0])
    else:
        top1_idx = np.argmax(probs, axis=0)
        top1 = probs[top1_idx, np.arange(probs.shape[1])]
        if probs.shape[0] > 1:
            part = np.partition(probs, -2, axis=0)
            top2 = part[-2, :]
        else:
            top2 = np.zeros(probs.shape[1])
    return top1, top2, top1_idx


def process_source_to_target_matrix(payload: Dict, job: Dict, chunk_label: str, args: argparse.Namespace) -> Tuple[pd.DataFrame, Dict]:
    matrix = np.clip(payload["matrix"], 0.0, None)
    source_cells = payload["source_cells"].astype(str)
    target_cells = payload["target_cells"].astype(str)
    row_sum = matrix.sum(axis=1)
    probs = matrix / (row_sum[:, None] + EPS)
    top1, top2, top1_idx = top2_from_probs(probs, axis=1)
    margin = top1 - top2
    ratio_log = np.log(top1 / (top2 + EPS) + EPS)
    entropy = -(probs * np.log(probs + EPS)).sum(axis=1)
    n_target = max(probs.shape[1], 2)
    entropy_conf = np.clip(1.0 - entropy / np.log(n_target), 0.0, 1.0)
    eff_n = np.exp(entropy)
    eff_conf = np.clip(1.0 - ((eff_n - 1.0) / max(n_target - 1.0, 1.0)), 0.0, 1.0)
    combined = (
        0.40 * percentile_rank(pd.Series(entropy_conf)).to_numpy()
        + 0.30 * percentile_rank(pd.Series(margin)).to_numpy()
        + 0.30 * percentile_rank(pd.Series(top1)).to_numpy()
    )

    edge_mask = probs >= args.edge_min_probability
    src_idx, tgt_idx = np.where(edge_mask)
    edges = pd.DataFrame(
        {
            "patient_id": job["patient_id"],
            "dataset_id": job["dataset_id"],
            "transition": job["transition"],
            "matrix_direction": "source_to_target",
            "matrix_storage": job["matrix_storage"],
            "chunk_label": chunk_label,
            "source_cell_moscot": source_cells[src_idx],
            "target_cell_moscot": target_cells[tgt_idx],
            "source_cell_original_obs": [strip_moscot_concat_suffix(x) for x in source_cells[src_idx]],
            "target_cell_original_obs": [strip_moscot_concat_suffix(x) for x in target_cells[tgt_idx]],
            "matrix_value": matrix[src_idx, tgt_idx],
            "ot_p_row_norm": probs[src_idx, tgt_idx],
            "source_target_rank": pd.Series(probs[src_idx, tgt_idx]).groupby(src_idx).rank(method="first", ascending=False).astype(int).to_numpy(),
            "source_is_top1_target": (tgt_idx == top1_idx[src_idx]).astype(int),
            "source_max_coupling": top1[src_idx],
            "source_second_coupling": top2[src_idx],
            "source_top12_margin": margin[src_idx],
            "source_top12_ratio_log": ratio_log[src_idx],
            "source_entropy_confidence": entropy_conf[src_idx],
            "source_effective_n_targets": eff_n[src_idx],
            "source_effective_target_confidence": eff_conf[src_idx],
            "source_retained_mass": row_sum[src_idx],
            "source_combined_concentration": combined[src_idx],
            "confidence_score": combined[src_idx],
        }
    )
    summary = {
        **job,
        "chunk_label": chunk_label,
        "n_source_cells": int(matrix.shape[0]),
        "n_target_cells": int(matrix.shape[1]),
        "n_output_edges": int(len(edges)),
        "edge_min_probability": float(args.edge_min_probability),
        "source_max_coupling_median": float(np.median(top1)) if len(top1) else np.nan,
        "source_entropy_confidence_median": float(np.median(entropy_conf)) if len(entropy_conf) else np.nan,
        "source_top12_margin_median": float(np.median(margin)) if len(margin) else np.nan,
    }
    return edges, summary


def process_target_to_source_matrix(payload: Dict, job: Dict, chunk_label: str, args: argparse.Namespace) -> Tuple[pd.DataFrame, Dict]:
    matrix = np.clip(payload["matrix"], 0.0, None)
    source_cells = payload["source_cells"].astype(str)
    target_cells = payload["target_cells"].astype(str)
    col_sum = matrix.sum(axis=0)
    probs = matrix / (col_sum[None, :] + EPS)
    top1, top2, top1_idx = top2_from_probs(probs, axis=0)
    margin = top1 - top2
    ratio_log = np.log(top1 / (top2 + EPS) + EPS)
    entropy = -(probs * np.log(probs + EPS)).sum(axis=0)
    n_source = max(probs.shape[0], 2)
    entropy_conf = np.clip(1.0 - entropy / np.log(n_source), 0.0, 1.0)
    eff_n = np.exp(entropy)
    eff_conf = np.clip(1.0 - ((eff_n - 1.0) / max(n_source - 1.0, 1.0)), 0.0, 1.0)
    combined = (
        0.40 * percentile_rank(pd.Series(entropy_conf)).to_numpy()
        + 0.30 * percentile_rank(pd.Series(margin)).to_numpy()
        + 0.30 * percentile_rank(pd.Series(top1)).to_numpy()
    )

    edge_mask = probs >= args.edge_min_probability
    src_idx, tgt_idx = np.where(edge_mask)
    edges = pd.DataFrame(
        {
            "patient_id": job["patient_id"],
            "dataset_id": job["dataset_id"],
            "transition": job["transition"],
            "matrix_direction": "target_to_source",
            "matrix_storage": job["matrix_storage"],
            "chunk_label": chunk_label,
            "source_cell_moscot": source_cells[src_idx],
            "target_cell_moscot": target_cells[tgt_idx],
            "source_cell_original_obs": [strip_moscot_concat_suffix(x) for x in source_cells[src_idx]],
            "target_cell_original_obs": [strip_moscot_concat_suffix(x) for x in target_cells[tgt_idx]],
            "matrix_value": matrix[src_idx, tgt_idx],
            "ot_p_col_norm": probs[src_idx, tgt_idx],
            "target_source_rank": pd.Series(probs[src_idx, tgt_idx]).groupby(tgt_idx).rank(method="first", ascending=False).astype(int).to_numpy(),
            "target_is_top1_source": (src_idx == top1_idx[tgt_idx]).astype(int),
            "target_max_coupling": top1[tgt_idx],
            "target_second_coupling": top2[tgt_idx],
            "target_top12_margin": margin[tgt_idx],
            "target_top12_ratio_log": ratio_log[tgt_idx],
            "target_entropy_confidence": entropy_conf[tgt_idx],
            "target_effective_n_sources": eff_n[tgt_idx],
            "target_effective_source_confidence": eff_conf[tgt_idx],
            "target_retained_mass": col_sum[tgt_idx],
            "target_combined_concentration": combined[tgt_idx],
            "confidence_score": combined[tgt_idx],
        }
    )
    summary = {
        **job,
        "chunk_label": chunk_label,
        "n_source_cells": int(matrix.shape[0]),
        "n_target_cells": int(matrix.shape[1]),
        "n_output_edges": int(len(edges)),
        "edge_min_probability": float(args.edge_min_probability),
        "target_max_coupling_median": float(np.median(top1)) if len(top1) else np.nan,
        "target_entropy_confidence_median": float(np.median(entropy_conf)) if len(entropy_conf) else np.nan,
        "target_top12_margin_median": float(np.median(margin)) if len(margin) else np.nan,
    }
    return edges, summary


def write_npz_analysis(args: argparse.Namespace) -> None:
    if args.npz_file is not None:
        write_single_npz_csv(args)
        return

    jobs = discover_npz_matrix_jobs(args.base_dir, args.matrix_direction)
    if args.limit_files is not None:
        jobs = jobs.head(args.limit_files).copy()
    if not args.write_to_dataset_dirs:
        ensure_clean_out_dir(args.out_dir, args.overwrite)
        jobs.to_csv(args.out_dir / "npz_matrix_jobs.tsv", sep="\t", index=False)
    if args.dry_run:
        print(jobs.to_string(index=False))
        print("Dry run only. No NPZ analysis written.")
        return

    sep = "," if args.analysis_format == "csv" else "\t"
    suffix = "csv" if args.analysis_format == "csv" else "tsv"
    edges_path = args.out_dir / f"ot_confidence_edges_from_npz.{suffix}"
    summary_path = args.out_dir / f"ot_confidence_summary_from_npz.{suffix}"
    summaries = []
    wrote_header = False

    for i, job in enumerate(jobs.to_dict(orient="records"), start=1):
        print("=" * 80)
        print(f"[npz {i}/{len(jobs)}] {job['patient_id']}/{job['dataset_id']} {job['transition']} {job['direction']} {job['matrix_storage']}")
        print("=" * 80)

        dataset_out_dir = Path(job["dataset_dir"]) / "OT_confidence_npz_csv"
        if args.write_to_dataset_dirs:
            dataset_out_dir.mkdir(parents=True, exist_ok=True)
            dataset_jobs_path = dataset_out_dir / "npz_matrix_jobs.csv"
            pd.DataFrame([job]).to_csv(
                dataset_jobs_path,
                index=False,
                mode="a",
                header=not dataset_jobs_path.exists(),
            )

        if job["matrix_storage"] == "full":
            payload = load_transport_npz(Path(job["npz_file"]))
            if job["direction"] == "source_to_target":
                edges, summary = process_source_to_target_matrix(payload, job, "full", args)
            else:
                edges, summary = process_target_to_source_matrix(payload, job, "full", args)
            if args.write_to_dataset_dirs:
                per_csv = dataset_out_dir / f"ot_confidence_{job['transition']}_{job['direction']}_full.csv"
                if per_csv.exists() and not args.overwrite:
                    raise FileExistsError(f"Output CSV exists. Use --overwrite or choose another output: {per_csv}")
                edges.to_csv(per_csv, index=False)
                pd.DataFrame([summary]).to_csv(per_csv.with_name(per_csv.stem + "_summary.csv"), index=False)
                print(f"wrote edges={len(edges):,} -> {per_csv}")
            else:
                edges.to_csv(edges_path, sep=sep, index=False, mode="a", header=not wrote_header)
                wrote_header = True
            summaries.append(summary)
            if not args.write_to_dataset_dirs:
                print(f"wrote edges={len(edges):,}")
            continue

        meta_path = Path(job["metadata_file"])
        meta = json.loads(meta_path.read_text())
        chunk_dir = meta_path.parent
        for chunk in meta.get("chunks", []):
            npz_path = chunk_dir / chunk["file"]
            payload = load_transport_npz(npz_path)
            chunk_label = f"chunk_{int(chunk.get('chunk_idx', len(summaries))):04d}"
            if job["direction"] == "source_to_target":
                edges, summary = process_source_to_target_matrix(payload, job, chunk_label, args)
            else:
                edges, summary = process_target_to_source_matrix(payload, job, chunk_label, args)
            if args.write_to_dataset_dirs:
                per_csv = dataset_out_dir / f"ot_confidence_{job['transition']}_{job['direction']}_{chunk_label}.csv"
                if per_csv.exists() and not args.overwrite:
                    raise FileExistsError(f"Output CSV exists. Use --overwrite or choose another output: {per_csv}")
                edges.to_csv(per_csv, index=False)
                pd.DataFrame([summary]).to_csv(per_csv.with_name(per_csv.stem + "_summary.csv"), index=False)
            else:
                edges.to_csv(edges_path, sep=sep, index=False, mode="a", header=not wrote_header)
                wrote_header = True
            summaries.append(summary)
            print(f"  {chunk_label}: wrote edges={len(edges):,}")
            del payload, edges
            gc.collect()

    if args.write_to_dataset_dirs:
        summary_all = pd.DataFrame(summaries)
        for dataset_dir, group in summary_all.groupby("dataset_dir", sort=False):
            out_dir = Path(dataset_dir) / "OT_confidence_npz_csv"
            group.to_csv(out_dir / "ot_confidence_summary_all.csv", index=False)
        print("=" * 80)
        print("NPZ analysis done")
        print("=" * 80)
        print("CSV outputs were written inside each PATIENT/DATASET/OT_confidence_npz_csv folder.")
        return

    pd.DataFrame(summaries).to_csv(summary_path, sep=sep, index=False)
    config = vars(args).copy()
    config["base_dir"] = str(args.base_dir)
    config["out_dir"] = str(args.out_dir)
    (args.out_dir / "npz_analysis_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print("=" * 80)
    print("NPZ analysis done")
    print("=" * 80)
    print(f"Edges:   {edges_path}")
    print(f"Summary: {summary_path}")


def write_single_npz_csv(args: argparse.Namespace) -> None:
    npz_path = Path(args.npz_file)
    if not npz_path.exists():
        raise FileNotFoundError(f"Missing NPZ file: {npz_path}")

    job = infer_job_from_npz_file(npz_path, args.matrix_direction)
    if args.output_csv is None:
        output_dir = Path(job["dataset_dir"]) / "OT_confidence_npz_csv"
        output_csv = output_dir / f"ot_confidence_{job['transition']}_{job['direction']}_{npz_path.stem}.csv"
    else:
        output_csv = Path(args.output_csv)
    if output_csv.exists() and not args.overwrite:
        raise FileExistsError(f"Output CSV exists. Use --overwrite or choose a new file: {output_csv}")
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    payload = load_transport_npz(npz_path)
    direction = job["direction"]
    if direction == "source_to_target":
        edges, summary = process_source_to_target_matrix(payload, job, npz_path.stem, args)
    elif direction == "target_to_source":
        edges, summary = process_target_to_source_matrix(payload, job, npz_path.stem, args)
    else:
        raise ValueError(f"Could not infer matrix direction. Use --matrix-direction: {direction}")

    if args.dry_run:
        print(json.dumps(summary, indent=2))
        print("Dry run only. No CSV written.")
        return

    edges.to_csv(output_csv, index=False)
    summary_csv = output_csv.with_name(output_csv.stem + "_summary.csv")
    pd.DataFrame([summary]).to_csv(summary_csv, index=False)
    print("=" * 80)
    print("Single NPZ CSV done")
    print("=" * 80)
    print(f"Input NPZ: {npz_path}")
    print(f"Edges CSV: {output_csv}")
    print(f"Summary CSV: {summary_csv}")


class H5ADCache:
    def __init__(self, base_dir: Path):
        self.base_dir = Path(base_dir)
        self.cache: Dict[str, ad.AnnData] = {}

    def get(self, rel_path: str) -> ad.AnnData:
        require_h5ad_stack()
        rel_path = clean_str(rel_path)
        if rel_path not in self.cache:
            path = self.base_dir / rel_path
            if not path.exists():
                raise FileNotFoundError(f"Missing h5ad: {path}")
            self.cache[rel_path] = ad.read_h5ad(path)
        return self.cache[rel_path]

    def clear(self) -> None:
        self.cache.clear()
        gc.collect()


def build_indexer(adata: ad.AnnData, obs_names: List[str]) -> np.ndarray:
    obs_to_idx = pd.Series(np.arange(adata.n_obs), index=adata.obs_names)
    idx = obs_to_idx.reindex(obs_names)
    if idx.isna().any():
        missing = idx[idx.isna()].index.tolist()[:10]
        raise KeyError(f"Missing obs_names in h5ad: examples={missing}")
    return idx.astype(int).to_numpy()


def extract_group_expression(group_df: pd.DataFrame, base_dir: Path, cache: H5ADCache) -> Tuple[sp.csr_matrix, sp.csr_matrix]:
    baseline_rel = group_df["baseline_h5ad"].iloc[0]
    target_rel = group_df["target_h5ad"].iloc[0]
    source_cells = group_df["source_cell_original_obs"].astype(str).tolist()
    target_cells = group_df["target_cell_original_obs"].astype(str).tolist()
    a_src = cache.get(baseline_rel)
    a_tgt = cache.get(target_rel)
    src_idx = build_indexer(a_src, source_cells)
    tgt_idx = build_indexer(a_tgt, target_cells)
    X_src = to_csr_matrix(a_src.X[src_idx, :])
    X_tgt = to_csr_matrix(a_tgt.X[tgt_idx, :])
    if X_src.shape != X_tgt.shape:
        raise ValueError(f"Source/target shapes differ: {X_src.shape} vs {X_tgt.shape}")
    return X_src, X_tgt


def extract_expression_for_pairs(pair_df: pd.DataFrame, base_dir: Path) -> Tuple[sp.csr_matrix, sp.csr_matrix]:
    require_h5ad_stack()
    cache = H5ADCache(base_dir)
    X_src_parts = []
    X_tgt_parts = []
    order_parts = []
    try:
        grouped = pair_df.groupby(["baseline_h5ad", "target_h5ad"], sort=False)
        for (baseline_rel, target_rel), group in grouped:
            print(f"    extracting group: {baseline_rel} -> {target_rel} rows={len(group):,}")
            X_src, X_tgt = extract_group_expression(group, base_dir, cache)
            X_src_parts.append(X_src)
            X_tgt_parts.append(X_tgt)
            order_parts.extend(group.index.tolist())
        X_src_all = sp.vstack(X_src_parts, format="csr")
        X_tgt_all = sp.vstack(X_tgt_parts, format="csr")
        restore = np.argsort(np.array(order_parts))
        return X_src_all[restore, :], X_tgt_all[restore, :]
    finally:
        cache.clear()


def get_gene_var_from_first_h5ad(base_dir: Path, pair_df: pd.DataFrame) -> pd.DataFrame:
    require_h5ad_stack()
    first_rel = pair_df["baseline_h5ad"].iloc[0]
    adata = ad.read_h5ad(base_dir / first_rel, backed="r")
    var = adata.var.copy()
    var.index = adata.var_names.copy()
    adata.file.close()
    return var


def write_shard(shard_df: pd.DataFrame, base_dir: Path, out_path: Path, var: pd.DataFrame, compression: Optional[str]) -> Dict:
    require_h5ad_stack()
    print(f"  writing shard: {out_path}")
    print(f"  shard rows: {len(shard_df):,}")
    X_src, X_tgt = extract_expression_for_pairs(shard_df, base_dir)
    obs = shard_df.copy()
    obs.index = obs["pair_id"].astype(str).tolist()
    obs = safe_obs_value_frame(obs)
    adata = ad.AnnData(X=X_src, obs=obs, var=var.copy())
    adata.layers["target"] = X_tgt
    adata.uns["paired_training_format"] = {
        "format": "paired_moscot_expression_h5ad",
        "row_unit": "one_high_confidence_moscot_paired_training_example",
        "X": "source_baseline_expression",
        "layers_target": "target_posttreatment_expression",
        "genes": "unified_zero_filled_ensembl_gene_order",
        "n_pairs": int(adata.n_obs),
        "n_genes": int(adata.n_vars),
    }
    if compression == "none" or compression is None:
        adata.write_h5ad(out_path)
    else:
        adata.write_h5ad(out_path, compression=compression)
    info = {
        "shard_file": str(out_path),
        "n_pairs": int(adata.n_obs),
        "n_genes": int(adata.n_vars),
        "source_nnz": int(X_src.nnz),
        "target_nnz": int(X_tgt.nnz),
        "file_size_gb": round(out_path.stat().st_size / 1e9, 4),
    }
    del adata, X_src, X_tgt
    gc.collect()
    return info


def write_shards(pair_df: pd.DataFrame, base_dir: Path, out_dir: Path, shard_size: int, compression: Optional[str]) -> pd.DataFrame:
    shards_dir = out_dir / "shards"
    shards_dir.mkdir(parents=True, exist_ok=True)
    var = get_gene_var_from_first_h5ad(base_dir, pair_df)
    genes_df = var.copy()
    genes_df.insert(0, "gene_id", genes_df.index.astype(str))
    genes_df.to_csv(out_dir / "genes.tsv", sep="\t", index=False)

    rows = []
    n_shards = (len(pair_df) + shard_size - 1) // shard_size
    for shard_idx in range(n_shards):
        start = shard_idx * shard_size
        end = min(start + shard_size, len(pair_df))
        shard_path = shards_dir / f"paired_expr_shard_{shard_idx:05d}.h5ad"
        print("=" * 80)
        print(f"Shard {shard_idx + 1}/{n_shards}: rows {start:,}-{end:,}")
        print("=" * 80)
        info = write_shard(pair_df.iloc[start:end].copy(), base_dir, shard_path, var, compression)
        info.update(
            {
                "shard_idx": int(shard_idx),
                "start_row": int(start),
                "end_row": int(end),
                "relative_shard_file": shard_path.relative_to(out_dir).as_posix(),
            }
        )
        rows.append(info)
    return pd.DataFrame(rows)


def ensure_clean_out_dir(out_dir: Path, overwrite: bool) -> None:
    if out_dir.exists() and overwrite:
        shutil.rmtree(out_dir)
    if out_dir.exists():
        raise FileExistsError(f"Output directory exists. Use --overwrite: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)


def write_global_summary(out_dir: Path, args: argparse.Namespace, scan_df: pd.DataFrame, valid_df: pd.DataFrame, pair_df: pd.DataFrame, manifest_df: pd.DataFrame) -> None:
    prob = pd.to_numeric(pair_df["probability"], errors="coerce")
    summary = {
        "base_dir": str(args.base_dir),
        "out_dir": str(out_dir),
        "confidence_metric": args.confidence_metric,
        "confidence_axis": args.confidence_axis,
        "pair_selection": args.pair_selection,
        "top_fraction": args.top_fraction if args.top_fraction is not None else "",
        "min_confidence": args.min_confidence if args.min_confidence is not None else "",
        "threshold_scope": "pair_file",
        "actual_pairs": int(len(pair_df)),
        "shard_size": int(args.shard_size),
        "n_shards": int(len(manifest_df)),
        "n_pair_files_scanned": int(len(scan_df)),
        "n_pair_files_valid": int(len(valid_df)),
        "unique_patients": int(pair_df["patient_id"].nunique()),
        "unique_datasets": int(pair_df[["patient_id", "dataset_id"]].drop_duplicates().shape[0]),
        "unique_transitions": int(pair_df["transition"].nunique()),
        "unique_source_cells": int(pair_df["source_cell_original_obs"].nunique()),
        "unique_target_cells": int(pair_df["target_cell_original_obs"].nunique()),
        "probability_mean": float(prob.mean()),
        "probability_median": float(prob.median()),
        "probability_min": float(prob.min()),
        "probability_max": float(prob.max()),
    }
    pd.DataFrame([summary]).to_csv(out_dir / "paired_h5ad_global_summary.tsv", sep="\t", index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir", type=Path, default=DEFAULT_BASE_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--npz-file",
        type=Path,
        default=None,
        help="Direct input NPZ file. If set, the script analyzes only this matrix file and ignores pair CSV files.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Direct output CSV path for --npz-file mode. If omitted, writes under the dataset folder containing the h5ad files.",
    )
    parser.add_argument(
        "--write-to-dataset-dirs",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="For folder iteration, write CSV files inside each PATIENT/DATASET folder beside the h5ad files.",
    )
    parser.add_argument(
        "--high-confidence-dir-name",
        type=str,
        default="OT_high_confidence",
        help="Parent output directory name created inside each PATIENT/DATASET folder.",
    )
    parser.add_argument(
        "--split-confidence-metrics",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write one output subdirectory and CSV set per confidence metric.",
    )
    parser.add_argument(
        "--metrics-to-split",
        type=str,
        default="max_coupling,entropy_confidence,top12_margin,top12_ratio_log,effective_target_confidence,retained_mass,combined_concentration",
        help="Comma-separated confidence metrics to split, or 'all'.",
    )
    parser.add_argument(
        "--output-mode",
        type=str,
        default="analysis_csv",
        choices=["npz_analysis", "analysis_csv", "paired_h5ad"],
        help="npz_analysis reads saved MOSCOT transport_matrix*.npz files; analysis_csv uses exported nonzero pair CSV; paired_h5ad builds model-training shards.",
    )
    parser.add_argument(
        "--analysis-format",
        type=str,
        default="csv",
        choices=["csv", "tsv"],
    )
    parser.add_argument(
        "--confidence-metric",
        type=str,
        default="combined_concentration",
        choices=CONFIDENCE_METRICS,
        help="Local confidence strategy used for the summary score. NPZ analysis computes metrics from each full matrix/chunk.",
    )
    parser.add_argument(
        "--confidence-axis",
        type=str,
        default="source",
        choices=["source", "target", "both"],
        help="source uses row-normalized source-to-target confidence; target uses column-normalized target-to-source view; both averages compatible combined metrics.",
    )
    parser.add_argument(
        "--matrix-direction",
        type=str,
        default="source_to_target",
        choices=["source_to_target", "target_to_source", "both"],
        help="Which saved MOSCOT NPZ direction to analyze. source_to_target chunks source rows; target_to_source chunks target columns.",
    )
    parser.add_argument(
        "--pair-direction",
        type=str,
        default="source_to_target",
        choices=["source_to_target", "target_to_source", "both"],
        help="Which saved nonzero pair CSV direction to analyze in analysis_csv mode.",
    )
    parser.add_argument(
        "--edge-min-probability",
        type=float,
        default=DEFAULT_EDGE_MIN_PROBABILITY,
        help="Only write matrix edges whose row/column-normalized probability is at least this value.",
    )
    parser.add_argument(
        "--confidence-column",
        type=str,
        default=None,
        help="Deprecated compatibility option. Use --confidence-metric instead.",
    )
    parser.add_argument(
        "--pair-selection",
        type=str,
        default="all",
        choices=["all", "top1", "top1_source", "top1_target", "mutual_top1"],
        help="For analysis use all. top1_source keeps strongest target per source; top1_target keeps strongest source per target; mutual_top1 keeps reciprocal top matches.",
    )
    parser.add_argument("--top-fraction", type=float, default=None)
    parser.add_argument("--min-confidence", type=float, default=None)
    parser.add_argument("--shard-size", type=int, default=25_000)
    parser.add_argument("--compression", type=str, default="gzip", choices=["gzip", "lzf", "none"])
    parser.add_argument("--limit-files", type=int, default=None)
    parser.add_argument("--no-match-check", action="store_true")
    parser.add_argument("--match-check-rows", type=int, default=1000)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.confidence_column is not None and args.confidence_metric == "combined_concentration":
        print("WARNING: --confidence-column is deprecated; using that value as --confidence-metric for compatibility.")
        if args.confidence_column not in CONFIDENCE_METRICS:
            raise ValueError(f"--confidence-column must be one of {CONFIDENCE_METRICS} when used as compatibility input.")
        args.confidence_metric = args.confidence_column
    if args.output_mode == "paired_h5ad" and args.top_fraction is None and args.min_confidence is None:
        raise ValueError("Provide --top-fraction, --min-confidence, or both.")
    if args.top_fraction is not None and not 0.0 < args.top_fraction <= 1.0:
        raise ValueError("--top-fraction must be in (0, 1].")
    if args.npz_file is None and not args.base_dir.exists():
        raise FileNotFoundError(f"Missing base dir: {args.base_dir}")

    print("=" * 80)
    print("Build local OT confidence outputs from full zero-filled data")
    print("=" * 80)
    print(f"Base dir:          {args.base_dir}")
    print(f"Out dir:           {args.out_dir}")
    print(f"Output mode:       {args.output_mode}")
    print(f"Analysis format:   {args.analysis_format}")
    print(f"Matrix direction:  {args.matrix_direction}")
    print(f"Pair direction:    {args.pair_direction}")
    print(f"Edge min prob:     {args.edge_min_probability}")
    print(f"Confidence metric: {args.confidence_metric}")
    print(f"Confidence axis:   {args.confidence_axis}")
    print(f"Pair selection:    {args.pair_selection}")
    print(f"Top fraction:      {args.top_fraction}")
    print(f"Min confidence:    {args.min_confidence}")
    print("Metric scope:      pair_file (patient/dataset/transition local)")
    print(f"Shard size:        {args.shard_size:,}")
    print(f"Dry run:           {args.dry_run}")
    print("=" * 80)

    if args.output_mode == "npz_analysis":
        write_npz_analysis(args)
        return

    scan_df, valid_df = scan_pair_files(
        args.base_dir,
        do_match_check=not args.no_match_check,
        match_check_rows=args.match_check_rows,
        pair_direction=args.pair_direction,
    )
    if args.limit_files is not None:
        valid_df = valid_df.head(args.limit_files).copy()

    if args.output_mode == "analysis_csv":
        if args.dry_run:
            print(valid_df.to_string(index=False))
            print("Dry run only. No analysis CSV written.")
            return
        write_analysis_csv(scan_df, valid_df, args)
        return

    chunks = []
    filter_rows = []
    running = 0
    for i, row in enumerate(valid_df.to_dict(orient="records"), start=1):
        print("=" * 80)
        print(f"[filter {i}/{len(valid_df)}] {row['relative_pair_file']}")
        print("=" * 80)
        filtered, summary = filter_one_pair_file(pd.Series(row), args, running)
        filter_rows.append(summary)
        if len(filtered) > 0:
            chunks.append(filtered)
            running += len(filtered)
        print(
            f"kept {summary['n_pairs_after']:,}/{summary['n_pairs_before']:,} "
            f"threshold={summary['threshold']:.6g}"
        )

    filter_summary = pd.DataFrame(filter_rows)
    if not chunks:
        raise ValueError("No pairs retained after filtering.")

    pair_df = pd.concat(chunks, axis=0, ignore_index=True)
    print("=" * 80)
    print(f"Retained pairs total: {len(pair_df):,}")
    print("=" * 80)

    if args.dry_run:
        print(filter_summary.to_string(index=False))
        print("Dry run only. No files written.")
        return

    ensure_clean_out_dir(args.out_dir, args.overwrite)
    scan_df.to_csv(args.out_dir / "paired_pair_file_scan_all.tsv", sep="\t", index=False)
    filter_summary.to_csv(args.out_dir / "filter_summary_by_pair_file.tsv", sep="\t", index=False)
    pair_df.to_csv(args.out_dir / "paired_moscot_pairs.tsv", sep="\t", index=False)

    manifest_df = write_shards(
        pair_df,
        args.base_dir,
        args.out_dir,
        args.shard_size,
        None if args.compression == "none" else args.compression,
    )
    manifest_df.to_csv(args.out_dir / "paired_h5ad_manifest.tsv", sep="\t", index=False)
    write_global_summary(args.out_dir, args, scan_df, valid_df, pair_df, manifest_df)

    config = vars(args).copy()
    config["base_dir"] = str(args.base_dir)
    config["out_dir"] = str(args.out_dir)
    config["threshold_scope"] = "pair_file"
    config["n_pairs_retained"] = int(len(pair_df))
    (args.out_dir / "filter_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    print("=" * 80)
    print("Done")
    print("=" * 80)
    print(f"Output dir:     {args.out_dir}")
    print(f"Pair TSV:       {args.out_dir / 'paired_moscot_pairs.tsv'}")
    print(f"Manifest:       {args.out_dir / 'paired_h5ad_manifest.tsv'}")
    print(f"Global summary: {args.out_dir / 'paired_h5ad_global_summary.tsv'}")
    print(f"Genes:          {args.out_dir / 'genes.tsv'}")
    print(f"Shards:         {args.out_dir / 'shards'}")


if __name__ == "__main__":
    main()
