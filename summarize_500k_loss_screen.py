#!/usr/bin/env python3
"""Print a simple ranked table for the 500k fold-0 Chreode loss screen."""

from __future__ import annotations

import csv
import glob
import json
import math
import os
from pathlib import Path


PREFIX = "checkpoints_500k_loss_screen_level5_hierarchical_dit_"
PATTERN = f"{PREFIX}*_fold0/val_epoch_losses.csv"


def as_float(row: dict[str, str], key: str) -> float:
    value = row.get(key, "")
    if value in ("", "nan", "NaN", "None"):
        return math.nan
    return float(value)


def loss_tag_from_folder(folder: str, row: dict[str, str]) -> str:
    if row.get("loss_tag"):
        return row["loss_tag"]
    name = os.path.basename(folder)
    if name.startswith(PREFIX):
        name = name[len(PREFIX) :]
    return name.rsplit("_fold", 1)[0]


def read_config(folder: str) -> dict:
    path = Path(folder) / "run_config.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    paths = sorted(glob.glob(PATTERN))
    if not paths:
        raise SystemExit(f"No loss-screen validation files found with pattern: {PATTERN}")

    rows = []
    skipped = []
    for path in paths:
        folder = os.path.dirname(path)
        with open(path, newline="", encoding="utf-8") as f:
            data = list(csv.DictReader(f))
        if not data:
            skipped.append((folder, "empty csv"))
            continue

        last = data[-1]
        final_epoch = int(float(last["epoch"]))
        if final_epoch != 49:
            skipped.append((folder, f"incomplete final_epoch={final_epoch}"))
            continue

        config = read_config(folder)
        best = min(data, key=lambda row: as_float(row, "loss_pair"))
        rows.append(
            {
                "tag": loss_tag_from_folder(folder, last),
                "lambda_mmd": config.get("lambda_mmd", ""),
                "lambda_w2": config.get("lambda_w2", ""),
                "lambda_drift": config.get("lambda_drift", ""),
                "lambda_down": config.get("lambda_down", ""),
                "best_epoch": int(float(best["epoch"])),
                "best_pair": as_float(best, "loss_pair"),
                "final_pair": as_float(last, "loss_pair"),
                "final_loss": as_float(last, "loss"),
                "final_mmd": as_float(last, "metric_mmd"),
                "final_w2": as_float(last, "metric_w2"),
                "final_drift": as_float(last, "metric_drift"),
                "final_down": as_float(last, "metric_down"),
                "folder": folder,
            }
        )

    rows.sort(key=lambda row: (row["best_pair"], row["final_pair"], row["final_mmd"], row["final_w2"]))

    fields = [
        "rank",
        "tag",
        "lambda_mmd",
        "lambda_w2",
        "lambda_drift",
        "lambda_down",
        "best_epoch",
        "best_pair",
        "final_pair",
        "final_loss",
        "final_mmd",
        "final_w2",
        "final_drift",
        "final_down",
    ]

    print(",".join(fields))
    for rank, row in enumerate(rows, 1):
        out = {
            "rank": rank,
            **row,
            "best_pair": f"{row['best_pair']:.8g}",
            "final_pair": f"{row['final_pair']:.8g}",
            "final_loss": f"{row['final_loss']:.8g}",
            "final_mmd": f"{row['final_mmd']:.8g}",
            "final_w2": f"{row['final_w2']:.8g}",
            "final_drift": f"{row['final_drift']:.8g}",
            "final_down": f"{row['final_down']:.8g}",
        }
        print(",".join(str(out[field]) for field in fields))

    print()
    print("TOP 5:")
    for rank, row in enumerate(rows[:5], 1):
        print(
            f"{rank}. {row['tag']} "
            f"mmd={row['lambda_mmd']} w2={row['lambda_w2']} drift={row['lambda_drift']} "
            f"best_pair={row['best_pair']:.8g} final_pair={row['final_pair']:.8g} "
            f"final_mmd={row['final_mmd']:.8g} final_w2={row['final_w2']:.8g}"
        )

    if skipped:
        print()
        print("SKIPPED:")
        for folder, reason in skipped:
            print(f"{folder}: {reason}")


if __name__ == "__main__":
    main()
