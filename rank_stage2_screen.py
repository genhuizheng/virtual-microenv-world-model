#!/usr/bin/env python3
"""Rank Stage-2 transition models from their training output.

Reads `val_epoch_losses.csv` from each checkpoint directory and reports the
best validation loss per configuration.

Selection metric
----------------
`val_loss_pair` -- weighted MSE between the predicted and true target latent --
is used for every configuration, including CellFlow runs trained with the
flow-matching objective. Their training loss is not comparable to a pair MSE,
but `loss_pair` is logged for every model regardless of what it optimised, so
it is the one column on which all levels can be ranked against each other.

The level-0 baseline
--------------------
Level 0 predicts no change (`z_pred == z_source`). Its `val_loss_pair` is
therefore the loss of doing nothing, and it is the number that matters most in
this table: a level that does not beat it has not learned a treatment effect,
whatever its rank. The script computes each model's improvement over that floor
explicitly rather than leaving it to be eyeballed.

Usage:
    python rank_stage2_screen.py --pattern 'checkpoints_s2_*'
    python rank_stage2_screen.py --pattern 'checkpoints_s2_*_fold*' --top 3
"""

from __future__ import annotations

import argparse
import glob
import os
import re
from collections import defaultdict

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--pattern", default="checkpoints_s2_*")
    p.add_argument("--top", type=int, default=3, help="How many to recommend promoting.")
    p.add_argument(
        "--metric",
        default="loss_pair",
        help="Column from val_epoch_losses.csv to rank on. Default loss_pair, the "
        "only objective every level is scored on.",
    )
    return p.parse_args()


def config_tag(directory: str) -> tuple[str, str]:
    """Split 'checkpoints_s2_<tag>_fold<N>' into (tag, fold)."""
    name = os.path.basename(directory.rstrip("/\\"))
    name = re.sub(r"^checkpoints_s2_", "", name)
    match = re.search(r"_fold(\d+)$", name)
    if match:
        return name[: match.start()], match.group(1)
    return name, "?"


def main() -> None:
    args = parse_args()
    directories = sorted(d for d in glob.glob(args.pattern) if os.path.isdir(d))
    if not directories:
        raise SystemExit(f"no directories matched {args.pattern!r}")

    grouped = defaultdict(list)
    skipped = []
    for directory in directories:
        path = os.path.join(directory, "val_epoch_losses.csv")
        if not os.path.exists(path):
            skipped.append((directory, "no val_epoch_losses.csv (still running or failed)"))
            continue
        try:
            table = pd.read_csv(path)
        except Exception as exc:  # noqa: BLE001
            skipped.append((directory, repr(exc)))
            continue
        if args.metric not in table.columns or table.empty:
            skipped.append((directory, f"no {args.metric!r} column"))
            continue
        tag, fold = config_tag(directory)
        best = table.loc[table[args.metric].idxmin()]
        grouped[tag].append(
            {
                "fold": fold,
                "level": best.get("level", "?"),
                "variant": best.get("variant", "?"),
                "loss_mode": best.get("loss_mode", "?"),
                "best": float(best[args.metric]),
                "final": float(table[args.metric].iloc[-1]),
                "epoch": int(best.get("epoch", -1)),
                "n_epochs": len(table),
                "mmd": float(best.get("metric_mmd", np.nan)),
                "w2": float(best.get("metric_w2", np.nan)),
            }
        )

    if skipped:
        print(f"skipped {len(skipped)} directory(ies):")
        for directory, why in skipped[:8]:
            print(f"  {os.path.basename(directory)}: {why}")
        print()

    if not grouped:
        raise SystemExit("nothing to rank")

    rows = []
    for tag, records in grouped.items():
        best = np.array([r["best"] for r in records], dtype=float)
        rows.append(
            {
                "tag": tag,
                "n_folds": len(records),
                "level": records[0]["level"],
                "variant": records[0]["variant"],
                "loss_mode": records[0]["loss_mode"],
                "best": float(best.mean()),
                "sd": float(best.std()),
                "epoch": records[0]["epoch"],
                "n_epochs": records[0]["n_epochs"],
                "mmd": float(np.nanmean([r["mmd"] for r in records])),
                "w2": float(np.nanmean([r["w2"] for r in records])),
            }
        )

    # The identity baseline is the reference point, not just another row.
    baseline = next(
        (r["best"] for r in rows if str(r["level"]) in {"0", "0.0"} or "identity" in r["tag"]),
        None,
    )

    rows.sort(key=lambda r: r["best"])

    print(f"ranking on val_{args.metric} (lower is better)")
    print(f"configs: {len(rows)}   directories: {len(directories)}\n")
    header = (
        f"{'#':>2} {'config':<22}{'lvl':>4}{'variant':>20}{'loss':>10}{'fold':>5}"
        f"{'best':>11}{'sd':>9}{'ep':>4}{'vs base':>10}"
    )
    print(header)
    print("-" * len(header))
    for i, r in enumerate(rows, 1):
        if baseline and baseline > 0:
            delta = 100.0 * (baseline - r["best"]) / baseline
            vs = f"{delta:+.1f}%"
        else:
            vs = "n/a"
        print(
            f"{i:>2} {r['tag']:<22}{str(r['level']):>4}{str(r['variant']):>20}"
            f"{str(r['loss_mode']):>10}{r['n_folds']:>5}"
            f"{r['best']:>11.6f}{r['sd']:>9.5f}{r['epoch']:>4}{vs:>10}"
        )

    print("\n'vs base' = improvement over the level-0 identity baseline (predicting no change).")
    if baseline is None:
        print("WARNING: no level-0 baseline found. Without it there is no way to tell")
        print("         whether any of these models learned a treatment effect at all.")
    else:
        print(f"baseline val_{args.metric} = {baseline:.6f}")
        beat = [r for r in rows if r["best"] < baseline]
        print(f"{len(beat)}/{len(rows)} configurations beat the baseline.")
        if len(beat) <= 1:
            print("  -> Almost nothing beats 'predict no change'. Treat any ranking among")
            print("     the rest as noise, and look at the data or the objective before")
            print("     interpreting architecture differences.")

    if any(r["n_folds"] == 1 for r in rows):
        print("\nNOTE: single-fold results. Small differences are not separable from fold")
        print("      noise -- use this to pick candidates, not a winner.")

    top = [r for r in rows if baseline is None or r["best"] < baseline][: max(args.top, 1)]
    if top:
        print(f"\nSuggested promotion to full cross-validation ({len(top)}):")
        for r in top:
            print(f"  {r['tag']:<22} val_{args.metric}={r['best']:.6f}  (level {r['level']}, {r['variant']})")


if __name__ == "__main__":
    main()
