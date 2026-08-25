#!/usr/bin/env python3
"""Rank Chreode loss-parameter screen runs from validation CSV files."""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
from collections import defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--glob",
        default="checkpoints_500k_loss*_level5_hierarchical_dit_*_fold*/val_epoch_losses.csv",
        help="Glob pattern for validation epoch CSV files.",
    )
    parser.add_argument("--out-csv", type=Path, default=Path("500k_loss_screen_summary.csv"))
    parser.add_argument("--required-final-epoch", type=int, default=49)
    parser.add_argument("--allow-incomplete", action="store_true")
    return parser.parse_args()


def as_float(row: dict[str, str], name: str) -> float:
    value = row.get(name, "")
    if value in ("", "nan", "NaN", "None"):
        return math.nan
    return float(value)


def infer_tag(folder: str, row: dict[str, str]) -> str:
    if row.get("loss_tag"):
        return row["loss_tag"]
    stem = os.path.basename(folder)
    if "_fold" in stem:
        stem = stem.rsplit("_fold", 1)[0]
    for prefix in (
        "checkpoints_500k_loss_screen_level5_hierarchical_dit_",
        "checkpoints_500k_loss_level5_hierarchical_dit_",
    ):
        if stem.startswith(prefix):
            return stem[len(prefix) :]
    return stem


def load_config(folder: str) -> dict:
    path = Path(folder) / "run_config.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def main() -> None:
    args = parse_args()
    paths = sorted(set(glob.glob(args.glob)))
    if not paths:
        raise SystemExit(f"No files matched: {args.glob}")

    runs = []
    for path in paths:
        folder = os.path.dirname(path)
        with open(path, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        if not rows:
            continue

        last = rows[-1]
        final_epoch = int(float(last["epoch"]))
        if final_epoch != args.required_final_epoch and not args.allow_incomplete:
            print(f"SKIP not complete: {folder}, final_epoch={final_epoch}")
            continue

        config = load_config(folder)
        tag = infer_tag(folder, last)
        fold = last.get("fold_index", config.get("fold_index", ""))
        best_row = min(rows, key=lambda row: as_float(row, "loss_pair"))

        runs.append(
            {
                "loss_tag": tag,
                "fold": str(fold),
                "folder": folder,
                "lambda_mmd": config.get("lambda_mmd", ""),
                "lambda_w2": config.get("lambda_w2", ""),
                "lambda_drift": config.get("lambda_drift", ""),
                "lambda_down": config.get("lambda_down", ""),
                "best_epoch": int(float(best_row["epoch"])),
                "best_pair": as_float(best_row, "loss_pair"),
                "best_mmd": as_float(best_row, "metric_mmd"),
                "best_w2": as_float(best_row, "metric_w2"),
                "final_epoch": final_epoch,
                "final_loss": as_float(last, "loss"),
                "final_pair": as_float(last, "loss_pair"),
                "final_mmd": as_float(last, "metric_mmd"),
                "final_w2": as_float(last, "metric_w2"),
                "final_drift": as_float(last, "metric_drift"),
                "final_down": as_float(last, "metric_down"),
            }
        )

    groups: dict[str, list[dict]] = defaultdict(list)
    for run in runs:
        groups[run["loss_tag"]].append(run)

    summary = []
    for tag, group in groups.items():
        mean = lambda key: sum(float(row[key]) for row in group) / len(group)
        first = group[0]
        summary.append(
            {
                "loss_tag": tag,
                "n_folds": len({row["fold"] for row in group}),
                "lambda_mmd": first["lambda_mmd"],
                "lambda_w2": first["lambda_w2"],
                "lambda_drift": first["lambda_drift"],
                "lambda_down": first["lambda_down"],
                "mean_best_pair": mean("best_pair"),
                "mean_best_mmd": mean("best_mmd"),
                "mean_best_w2": mean("best_w2"),
                "mean_final_loss": mean("final_loss"),
                "mean_final_pair": mean("final_pair"),
                "mean_final_mmd": mean("final_mmd"),
                "mean_final_w2": mean("final_w2"),
                "mean_final_drift": mean("final_drift"),
                "mean_final_down": mean("final_down"),
            }
        )

    summary.sort(key=lambda row: (row["mean_best_pair"], row["mean_final_pair"], row["mean_final_mmd"], row["mean_final_w2"]))

    fields = [
        "rank",
        "loss_tag",
        "n_folds",
        "lambda_mmd",
        "lambda_w2",
        "lambda_drift",
        "lambda_down",
        "mean_best_pair",
        "mean_best_mmd",
        "mean_best_w2",
        "mean_final_loss",
        "mean_final_pair",
        "mean_final_mmd",
        "mean_final_w2",
        "mean_final_drift",
        "mean_final_down",
    ]

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for rank, row in enumerate(summary, 1):
            writer.writerow({"rank": rank, **row})

    print(",".join(fields))
    for rank, row in enumerate(summary, 1):
        print(",".join(str({"rank": rank, **row}[field]) for field in fields))

    print("\nTOP SETTINGS:")
    for rank, row in enumerate(summary[:5], 1):
        print(
            f"{rank}. {row['loss_tag']} "
            f"mmd={row['lambda_mmd']} w2={row['lambda_w2']} drift={row['lambda_drift']} "
            f"best_pair={row['mean_best_pair']:.8g} final_pair={row['mean_final_pair']:.8g} "
            f"final_mmd={row['mean_final_mmd']:.8g} final_w2={row['mean_final_w2']:.8g}"
        )

    print(f"\nwrote: {args.out_csv}")


if __name__ == "__main__":
    main()
