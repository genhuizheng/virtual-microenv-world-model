#!/usr/bin/env python3
"""Build a baseline-only spatial prediction manifest without reading expression matrices."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

from RNA_validation.common import clean, require_empty_output, write_json


def slug(value: object) -> str:
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", clean(value)).strip("_.")
    return text or "unknown"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--spatial-root", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--min-genes", type=int, default=10000)
    p.add_argument("--datasets", nargs="*", default=[])
    p.add_argument("--include-noninteger", action="store_true")
    p.add_argument("--max-samples", type=int, default=-1)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def is_baseline(value: object) -> bool:
    return clean(value).lower() == "baseline"


def main() -> None:
    args = parse_args()
    out = require_empty_output(args.output_dir, args.overwrite)
    tables = args.spatial_root / "all_dataset_statistics" / "tables"
    stats_path = tables / "all_h5ad_sample_statistics.csv"
    schema_path = tables / "schema_validation.csv"
    stats = pd.read_csv(stats_path, dtype=str, keep_default_na=False)
    schema = pd.read_csv(schema_path, dtype=str, keep_default_na=False)
    stats["n_genes_numeric"] = pd.to_numeric(stats["n_genes"], errors="coerce")
    eligible = stats.loc[stats["n_genes_numeric"] > args.min_genes].copy()
    if args.datasets:
        eligible = eligible.loc[eligible["gse_id"].isin(args.datasets)].copy()
    eligible = eligible.merge(schema[["file", "X_integer", "spatial_present", "spatial_shape_valid"]], on="file", how="left")
    eligible["eligible_reason"] = "eligible"
    eligible.loc[eligible["spatial_present"] != "True", "eligible_reason"] = "missing_spatial"
    eligible.loc[eligible["spatial_shape_valid"] != "True", "eligible_reason"] = "invalid_spatial_shape"
    if not args.include_noninteger:
        eligible.loc[eligible["X_integer"] != "True", "eligible_reason"] = "noninteger_expression"
    accepted = eligible.loc[eligible["eligible_reason"] == "eligible"].copy()

    baselines = []
    accepted = accepted.loc[accepted["timepoint"].map(is_baseline)].copy()
    for row in accepted.itertuples(index=False):
        sample_id = slug(f"{row.gse_id}__{row.patient_id}__{Path(row.file_name).stem}")
        baselines.append({
            "sample_id": sample_id,
            "gse_id": row.gse_id,
            "patient_key": row.patient_key,
            "patient_id": row.patient_id,
            "platform": row.platform,
            "cancer_type": row.cancer_type,
            "response_group": clean(row.response_group) or "NA",
            "original_response_status": clean(row.original_response_status) or "NA",
            "timepoint": row.timepoint,
            "pre_file": str((args.spatial_root / row.file).resolve()),
            "pre_n_obs": int(row.n_cells),
            "pre_n_genes": int(row.n_genes),
        })
    pair_table = pd.DataFrame(baselines)
    if pair_table.empty:
        raise ValueError("No baseline samples were found after filtering")
    if args.max_samples > 0:
        pair_table = pair_table.iloc[: args.max_samples].copy()
    pair_table.to_csv(out / "spatial_baselines.tsv", sep="\t", index=False)
    eligible.to_csv(out / "sample_eligibility.tsv", sep="\t", index=False)
    write_json(out / "manifest_summary.json", {
        "spatial_root": str(args.spatial_root),
        "strict_gene_rule": f"n_genes > {args.min_genes}",
        "integer_expression_required": not args.include_noninteger,
        "eligible_samples": int(len(accepted)),
        "baseline_samples": int(len(pair_table)),
        "patients": int(pair_table["patient_key"].nunique()),
        "datasets": sorted(pair_table["gse_id"].unique().tolist()),
        "prediction_input": "baseline H5AD only",
        "measured_post_treatment_usage": "none",
    })
    print(f"baselines={len(pair_table)} patients={pair_table['patient_key'].nunique()} datasets={pair_table['gse_id'].nunique()}")


if __name__ == "__main__":
    main()
