#!/usr/bin/env python3
"""
Filter paired h5ad training data to high-confidence OT/MOSCOT pairs.

Expected input directory should be built from the original unified zero-filled
paired data, not from a small random sample:

    paired_training_h5ad_500k_fast/
    paired_training_h5ad_full_zero_fill/
    ...

with:

    paired_h5ad_manifest.tsv
    paired_moscot_pairs.tsv
    genes.tsv
    shards/*.h5ad

The script filters rows inside each paired h5ad shard, writes a new paired h5ad
directory with the same training layout, and records the filtering settings.

Typical first use:

    python "initial biological analysis/filter_high_confidence_pairs.py" \
      --input-dir paired_training_h5ad_500k_fast \
      --output-dir paired_training_h5ad_500k_high_conf_top25pct \
      --confidence-column probability \
      --top-fraction 0.25 \
      --threshold-scope local \
      --group-column dataset_id
"""

from __future__ import annotations

from pathlib import Path
import argparse
import json
import shutil
from typing import Iterable

import anndata as ad
import numpy as np
import pandas as pd


EPS = 1e-8


COMBINED_COMPONENTS = {
    "entropy_confidence": 0.30,
    "top12_margin": 0.20,
    "cost_confidence": 0.20,
    "stability_confidence": 0.30,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a high-confidence filtered paired h5ad training directory."
    )
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--confidence-column", type=str, default="probability")
    parser.add_argument("--min-confidence", type=float, default=None)
    parser.add_argument("--top-fraction", type=float, default=None)
    parser.add_argument(
        "--threshold-scope",
        choices=["local", "global"],
        default="local",
        help=(
            "Use local to compute top-fraction thresholds within each group, "
            "usually each biological dataset. Use global only for diagnostics."
        ),
    )
    parser.add_argument(
        "--group-column",
        type=str,
        default="dataset_id",
        help="Column used for local thresholds. Default: dataset_id.",
    )
    parser.add_argument(
        "--compute-combined-confidence",
        action="store_true",
        help=(
            "If component columns are available, compute combined_confidence = "
            "0.30 entropy + 0.20 margin + 0.20 cost + 0.30 stability."
        ),
    )
    parser.add_argument("--combined-column", type=str, default="combined_confidence")
    parser.add_argument("--training-weight-gamma", type=float, default=1.0)
    parser.add_argument("--training-weight-min", type=float, default=0.20)
    parser.add_argument("--training-weight-max", type=float, default=3.00)
    parser.add_argument("--probability-col", type=str, default="probability")
    parser.add_argument("--manifest-name", type=str, default="paired_h5ad_manifest.tsv")
    parser.add_argument("--pairs-name", type=str, default="paired_moscot_pairs.tsv")
    parser.add_argument("--genes-name", type=str, default="genes.tsv")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")


def numeric_series(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        raise ValueError(f"Missing column '{column}'. Available columns: {list(frame.columns)}")
    return pd.to_numeric(frame[column], errors="coerce").replace([np.inf, -np.inf], np.nan)


def percentile_rank(values: pd.Series) -> pd.Series:
    valid = values.notna()
    out = pd.Series(np.nan, index=values.index, dtype=np.float64)
    if valid.any():
        out.loc[valid] = values.loc[valid].rank(method="average", pct=True)
    return out


def add_combined_confidence(obs: pd.DataFrame, combined_column: str) -> pd.DataFrame:
    missing = [col for col in COMBINED_COMPONENTS if col not in obs.columns]
    if missing:
        raise ValueError(
            "Cannot compute combined confidence. Missing component columns: "
            + ", ".join(missing)
        )

    combined = pd.Series(0.0, index=obs.index, dtype=np.float64)
    for col, weight in COMBINED_COMPONENTS.items():
        values = numeric_series(obs, col).clip(lower=0.0, upper=1.0).fillna(0.0)
        combined += weight * values
    obs = obs.copy()
    obs[combined_column] = combined.clip(lower=0.0, upper=1.0)
    return obs


def add_training_weight(
    obs: pd.DataFrame,
    confidence_column: str,
    gamma: float,
    weight_min: float,
    weight_max: float,
) -> pd.DataFrame:
    confidence = numeric_series(obs, confidence_column).fillna(0.0).clip(lower=0.0)
    raw = np.power(confidence.to_numpy(np.float64) + EPS, gamma)
    mean = float(np.mean(raw)) if len(raw) else 1.0
    if mean <= EPS:
        weight = np.ones_like(raw)
    else:
        weight = raw / mean
    weight = np.clip(weight, weight_min, weight_max)
    obs = obs.copy()
    obs["confidence_training_weight"] = weight.astype(np.float32)
    return obs


def collect_confidence_table(
    shard_paths: Iterable[Path],
    confidence_column: str,
    group_column: str,
    compute_combined_confidence: bool,
    combined_column: str,
) -> pd.DataFrame:
    rows = []
    for shard_path in shard_paths:
        adata = ad.read_h5ad(shard_path, backed="r")
        obs = adata.obs.copy()
        adata.file.close()
        if compute_combined_confidence:
            obs = add_combined_confidence(obs, combined_column)
        confidence = numeric_series(obs, confidence_column)
        if group_column in obs.columns:
            group = obs[group_column].astype(str)
        else:
            group = pd.Series("__all__", index=obs.index)
        rows.append(pd.DataFrame({"confidence": confidence, "group": group}))

    if not rows:
        return pd.DataFrame(columns=["confidence", "group"])
    return pd.concat(rows, ignore_index=True)


def compute_thresholds(
    confidence_table: pd.DataFrame,
    top_fraction: float | None,
    threshold_scope: str,
) -> tuple[float | None, dict[str, float]]:
    if top_fraction is None:
        return None, {}
    if not 0.0 < top_fraction <= 1.0:
        raise ValueError("--top-fraction must be in (0, 1].")

    finite = confidence_table.loc[np.isfinite(confidence_table["confidence"].to_numpy()), ["confidence", "group"]]
    if finite.empty:
        raise ValueError("No finite confidence values found.")

    quantile = 1.0 - top_fraction
    if threshold_scope == "global":
        return float(np.quantile(finite["confidence"], quantile)), {}

    thresholds = {}
    for group, frame in finite.groupby("group", sort=True):
        thresholds[str(group)] = float(np.quantile(frame["confidence"], quantile))
    return None, thresholds


def summarize_thresholds(threshold_by_group: dict[str, float]) -> list[dict]:
    return [
        {"group": group, "threshold": threshold}
        for group, threshold in sorted(threshold_by_group.items(), key=lambda item: item[0])
    ]


def keep_by_threshold(
    obs: pd.DataFrame,
    confidence: pd.Series,
    min_confidence: float | None,
    global_threshold: float | None,
    threshold_by_group: dict[str, float],
    group_column: str,
) -> pd.Series:
    threshold = min_confidence
    if global_threshold is not None:
        threshold = global_threshold if threshold is None else max(threshold, global_threshold)
    if threshold_by_group:
        if group_column in obs.columns:
            group = obs[group_column].astype(str)
        else:
            group = pd.Series("__all__", index=obs.index)
        local_threshold = group.map(threshold_by_group)
        if min_confidence is not None:
            local_threshold = np.maximum(local_threshold.astype(float), float(min_confidence))
        return confidence >= local_threshold

    if threshold is None:
        raise ValueError("Provide --min-confidence, --top-fraction, or both.")
    return confidence >= threshold
    if len(all_values) == 0:
        raise ValueError(f"No finite values found in confidence column '{confidence_column}'.")

    quantile = 1.0 - top_fraction
    return float(np.quantile(all_values, quantile))


def filter_pairs_table(
    pairs_path: Path,
    output_pairs_path: Path,
    kept_pair_ids: set[str],
    kept_source_target: set[tuple[str, str]],
) -> int:
    if not pairs_path.exists():
        return 0

    pairs = pd.read_csv(pairs_path, sep="\t")
    if pairs.empty:
        pairs.to_csv(output_pairs_path, sep="\t", index=False)
        return 0

    keep = pd.Series(False, index=pairs.index)
    if "pair_id" in pairs.columns and kept_pair_ids:
        keep |= pairs["pair_id"].astype(str).isin(kept_pair_ids)

    source_cols = ["source_cell_original_obs", "source_cell", "source_cell_id"]
    target_cols = ["target_cell_original_obs", "target_cell", "target_cell_id"]
    source_col = next((col for col in source_cols if col in pairs.columns), None)
    target_col = next((col for col in target_cols if col in pairs.columns), None)
    if source_col and target_col and kept_source_target:
        keys = list(zip(pairs[source_col].astype(str), pairs[target_col].astype(str)))
        keep |= pd.Series([key in kept_source_target for key in keys], index=pairs.index)

    filtered = pairs.loc[keep].copy()
    filtered.to_csv(output_pairs_path, sep="\t", index=False)
    return len(filtered)


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir
    output_dir = args.output_dir

    manifest_path = input_dir / args.manifest_name
    require_file(manifest_path)

    if output_dir.exists():
        if not args.overwrite and not args.dry_run:
            raise FileExistsError(f"Output directory exists. Use --overwrite: {output_dir}")
        if args.overwrite and not args.dry_run:
            shutil.rmtree(output_dir)

    manifest = pd.read_csv(manifest_path, sep="\t")
    if "relative_shard_file" not in manifest.columns:
        raise ValueError("Manifest must contain relative_shard_file column.")

    shard_paths = [input_dir / rel for rel in manifest["relative_shard_file"].astype(str)]
    for shard_path in shard_paths:
        require_file(shard_path)

    confidence_column = args.combined_column if args.compute_combined_confidence else args.confidence_column
    confidence_table = collect_confidence_table(
        shard_paths,
        confidence_column,
        args.group_column,
        args.compute_combined_confidence,
        args.combined_column,
    )
    global_threshold, threshold_by_group = compute_thresholds(
        confidence_table,
        args.top_fraction,
        args.threshold_scope,
    )
    if args.min_confidence is None and global_threshold is None and not threshold_by_group:
        raise ValueError("Provide --min-confidence, --top-fraction, or both.")

    summary_rows = []
    new_manifest_rows = []
    kept_pair_ids: set[str] = set()
    kept_source_target: set[tuple[str, str]] = set()

    for _, row in manifest.iterrows():
        rel = str(row["relative_shard_file"])
        shard_path = input_dir / rel
        adata = ad.read_h5ad(shard_path)
        obs = adata.obs.copy()

        if args.compute_combined_confidence:
            obs = add_combined_confidence(obs, args.combined_column)
            adata.obs = obs

        obs = add_training_weight(
            obs,
            confidence_column,
            args.training_weight_gamma,
            args.training_weight_min,
            args.training_weight_max,
        )
        adata.obs = obs

        confidence = numeric_series(obs, confidence_column)
        keep_mask = keep_by_threshold(
            obs,
            confidence,
            args.min_confidence,
            global_threshold,
            threshold_by_group,
            args.group_column,
        )
        keep_mask = keep_mask.fillna(False).to_numpy(bool)
        kept = int(keep_mask.sum())
        total = int(len(keep_mask))
        groups = obs[args.group_column].astype(str) if args.group_column in obs.columns else pd.Series("__all__", index=obs.index)

        summary_rows.append(
            {
                "relative_shard_file": rel,
                "groups": ",".join(sorted(groups.unique())),
                "n_pairs_before": total,
                "n_pairs_after": kept,
                "retained_fraction": kept / total if total else 0.0,
                "confidence_min_before": float(np.nanmin(confidence)) if total else np.nan,
                "confidence_median_before": float(np.nanmedian(confidence)) if total else np.nan,
                "confidence_max_before": float(np.nanmax(confidence)) if total else np.nan,
            }
        )

        if kept == 0:
            continue

        filtered = adata[keep_mask, :].copy()

        if "pair_id" in filtered.obs.columns:
            kept_pair_ids.update(filtered.obs["pair_id"].astype(str).tolist())
        if "source_cell_original_obs" in filtered.obs.columns and "target_cell_original_obs" in filtered.obs.columns:
            kept_source_target.update(
                zip(
                    filtered.obs["source_cell_original_obs"].astype(str),
                    filtered.obs["target_cell_original_obs"].astype(str),
                )
            )

        new_rel = Path("shards") / Path(rel).name
        new_manifest_row = row.copy()
        new_manifest_row["relative_shard_file"] = new_rel.as_posix()
        if "n_pairs" in new_manifest_row.index:
            new_manifest_row["n_pairs"] = kept
        new_manifest_rows.append(new_manifest_row)

        if not args.dry_run:
            out_shard = output_dir / new_rel
            out_shard.parent.mkdir(parents=True, exist_ok=True)
            filtered.write_h5ad(out_shard)

    summary = pd.DataFrame(summary_rows)
    new_manifest = pd.DataFrame(new_manifest_rows)

    config = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "confidence_column": confidence_column,
        "min_confidence": args.min_confidence,
        "top_fraction": args.top_fraction,
        "threshold_scope": args.threshold_scope,
        "group_column": args.group_column,
        "effective_global_threshold": global_threshold,
        "effective_group_thresholds": threshold_by_group,
        "compute_combined_confidence": args.compute_combined_confidence,
        "combined_components": COMBINED_COMPONENTS if args.compute_combined_confidence else None,
        "training_weight_gamma": args.training_weight_gamma,
        "training_weight_clip": [args.training_weight_min, args.training_weight_max],
        "n_pairs_before": int(summary["n_pairs_before"].sum()) if not summary.empty else 0,
        "n_pairs_after": int(summary["n_pairs_after"].sum()) if not summary.empty else 0,
    }
    config["retained_fraction"] = (
        config["n_pairs_after"] / config["n_pairs_before"] if config["n_pairs_before"] else 0.0
    )

    print(json.dumps(config, indent=2))

    if args.dry_run:
        print("Dry run only. No files written.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    new_manifest.to_csv(output_dir / args.manifest_name, sep="\t", index=False)
    summary.to_csv(output_dir / "filter_summary.tsv", sep="\t", index=False)
    if threshold_by_group:
        pd.DataFrame(summarize_thresholds(threshold_by_group)).to_csv(
            output_dir / "filter_thresholds_by_group.tsv",
            sep="\t",
            index=False,
        )
    (output_dir / "filter_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    for name in [args.genes_name, "paired_h5ad_global_summary.tsv"]:
        src = input_dir / name
        if src.exists():
            shutil.copy2(src, output_dir / name)

    filtered_pairs_count = filter_pairs_table(
        input_dir / args.pairs_name,
        output_dir / args.pairs_name,
        kept_pair_ids,
        kept_source_target,
    )
    if filtered_pairs_count == 0 and (input_dir / args.pairs_name).exists():
        print(
            "Warning: paired_moscot_pairs.tsv was present but no rows were matched. "
            "Check pair_id/source-target column names."
        )


if __name__ == "__main__":
    main()
