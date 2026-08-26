#!/usr/bin/env python3
"""Rank Stage-2 transition models from their training output.

Reads `val_epoch_losses.csv` from each checkpoint directory and reports the
best validation result per configuration.

Ranking within an arm
---------------------
Each configuration is ranked on the objective it actually optimised:

    pair_mse -> loss_pair    (weighted MSE against the paired target latent)
    chreode  -> loss         (MMD + Sinkhorn W2 + drift + downhill)
    cellflow -> loss         (flow-matching velocity MSE)

Ranking a chreode run on loss_pair would score it on the very target it was
built to avoid, so arms are ranked separately and never pooled into one column.

The level-0 floor, per arm
--------------------------
Level 0 predicts no change (`z_pred == z_source`). Each arm therefore has its
own floor and they mean different things:

    pair_mse floor  = the loss of doing nothing
    chreode  floor  = the MMD/W2 between the pre and post distributions as they
                      stand -- a much HARDER floor, because an identity map
                      already has substantial population overlap

That asymmetry is the point. A model that beats the floor on pair_mse but not
on chreode has learned a mean shift, not a treatment effect: it moved every
cell toward the average post-treatment state, which lowers per-cell MSE while
leaving (or making worse) the match between the predicted and true
distributions.

Cross-arm comparison
--------------------
`metric_mmd` and `metric_w2` are logged for every configuration regardless of
what it optimised, so they are the one basis on which the two arms can be read
against each other -- and they are not capped by MOSCOT pairing noise the way
loss_pair is.

Those are reported AT THE SELECTED CHECKPOINT, not at whichever epoch happened
to minimise MMD. Reporting the per-epoch minimum would credit each model for an
epoch its own selection rule would never have chosen, which is cherry-picking:
the deployed model is the selected one.

Usage:
    python rank_stage2_screen.py --pattern 'checkpoints_s2_500k_*'
    python rank_stage2_screen.py --pattern 'checkpoints_s2_*' --top 3
"""

from __future__ import annotations

import argparse
import glob
import os
import re
from collections import defaultdict

import numpy as np
import pandas as pd

# The column each loss mode is both optimised and ranked on.
ARM_METRIC = {
    "pair_mse": "loss_pair",
    "chreode": "loss",
    "cellflow": "loss",
}
DEFAULT_METRIC = "loss_pair"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--pattern", default="checkpoints_s2_*")
    p.add_argument("--top", type=int, default=3, help="How many to recommend promoting.")
    p.add_argument(
        "--metric",
        default=None,
        help="Override the per-arm ranking column. By default each arm is ranked "
        "on the objective it optimised (see ARM_METRIC).",
    )
    return p.parse_args()


def config_tag(directory: str) -> tuple[str, str]:
    """Split 'checkpoints_s2[_500k]_<tag>_fold<N>' into (tag, fold)."""
    name = os.path.basename(directory.rstrip("/\\"))
    name = re.sub(r"^checkpoints_s2_(500k_)?", "", name)
    match = re.search(r"_fold(\d+)$", name)
    if match:
        return name[: match.start()], match.group(1)
    return name, "?"


def is_level_zero(record: dict) -> bool:
    return str(record["level"]) in {"0", "0.0"} or "identity" in record["tag"]


def collect(pattern: str, metric_override: str | None):
    directories = sorted(d for d in glob.glob(pattern) if os.path.isdir(d))
    if not directories:
        raise SystemExit(f"no directories matched {pattern!r}")

    grouped: dict[str, list[dict]] = defaultdict(list)
    skipped: list[tuple[str, str]] = []

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
        if table.empty:
            skipped.append((directory, "empty val_epoch_losses.csv"))
            continue

        loss_mode = str(table.get("loss_mode", pd.Series(["?"])).iloc[0])
        metric = metric_override or ARM_METRIC.get(loss_mode, DEFAULT_METRIC)
        if metric not in table.columns:
            skipped.append((directory, f"no {metric!r} column"))
            continue

        tag, fold = config_tag(directory)
        best = table.loc[table[metric].idxmin()]
        grouped[tag].append(
            {
                "tag": tag,
                "fold": fold,
                "arm": loss_mode,
                "metric": metric,
                "level": best.get("level", "?"),
                "variant": best.get("variant", "?"),
                "best": float(best[metric]),
                "epoch": int(best.get("epoch", -1)),
                "n_epochs": len(table),
                # At the SELECTED epoch, not the per-epoch minimum.
                "mmd": float(best.get("metric_mmd", np.nan)),
                "w2": float(best.get("metric_w2", np.nan)),
                "loss_pair": float(best.get("loss_pair", np.nan)),
            }
        )

    return grouped, skipped, directories


def aggregate(grouped: dict[str, list[dict]]) -> list[dict]:
    rows = []
    for tag, records in grouped.items():
        best = np.array([r["best"] for r in records], dtype=float)
        rows.append(
            {
                "tag": tag,
                "n_folds": len(records),
                "arm": records[0]["arm"],
                "metric": records[0]["metric"],
                "level": records[0]["level"],
                "variant": records[0]["variant"],
                "best": float(best.mean()),
                "sd": float(best.std()),
                "epoch": records[0]["epoch"],
                "n_epochs": records[0]["n_epochs"],
                "mmd": float(np.nanmean([r["mmd"] for r in records])),
                "w2": float(np.nanmean([r["w2"] for r in records])),
                "loss_pair": float(np.nanmean([r["loss_pair"] for r in records])),
            }
        )
    return rows


def print_arm(arm: str, rows: list[dict], top: int) -> list[dict]:
    metric = rows[0]["metric"]
    rows = sorted(rows, key=lambda r: r["best"])
    floor = next((r["best"] for r in rows if is_level_zero(r)), None)

    print(f"\n=== arm: {arm}   (ranked on val_{metric}, lower is better) ===")
    header = (
        f"{'#':>2} {'config':<24}{'lvl':>4}{'variant':>20}{'fold':>5}"
        f"{'best':>11}{'sd':>9}{'ep':>4}{'/of':>4}{'vs floor':>10}{'mmd':>10}"
    )
    print(header)
    print("-" * len(header))
    for i, r in enumerate(rows, 1):
        if floor and floor > 0:
            vs = f"{100.0 * (floor - r['best']) / floor:+.1f}%"
        else:
            vs = "n/a"
        mmd = "  n/a" if np.isnan(r["mmd"]) else f"{r['mmd']:.5f}"
        print(
            f"{i:>2} {r['tag']:<24}{str(r['level']):>4}{str(r['variant']):>20}"
            f"{r['n_folds']:>5}{r['best']:>11.6f}{r['sd']:>9.5f}"
            f"{r['epoch']:>4}{r['n_epochs']:>4}{vs:>10}{mmd:>10}"
        )

    if floor is None:
        print("\n  WARNING: no level-0 floor in this arm. Without it there is no way to")
        print("           tell whether these models learned a treatment effect at all.")
        return rows[:top]

    print(f"\n  floor (level 0) val_{metric} = {floor:.6f}")
    beat = [r for r in rows if r["best"] < floor and not is_level_zero(r)]
    learned = [r for r in rows if not is_level_zero(r)]
    print(f"  {len(beat)}/{len(learned)} configurations beat the floor.")
    if len(beat) <= 1:
        print("  -> Almost nothing beats 'predict no change'. Treat any ranking among")
        print("     the rest as noise, and look at the data or the objective before")
        print("     interpreting architecture differences.")

    # Best epoch at or near 0 means the model stopped generalising immediately;
    # its rank then reflects what is learnable in one pass, not architecture.
    early = [r for r in learned if 0 <= r["epoch"] <= 1 and r["n_epochs"] > 3]
    if len(early) >= max(2, len(learned) // 2):
        print(f"  -> {len(early)}/{len(learned)} configs peaked at epoch 0-1 of "
              f"{rows[0]['n_epochs']}. Validation stopped improving almost")
        print("     immediately, so these ranks measure the one-pass solution, not")
        print("     capacity. Check the train/val gap before promoting anything.")

    return [r for r in beat][:top]


def print_cross_arm(rows: list[dict]) -> None:
    scored = [r for r in rows if not np.isnan(r["mmd"])]
    if len(scored) < 2 or len({r["arm"] for r in scored}) < 2:
        return
    scored.sort(key=lambda r: r["mmd"])

    print("\n=== cross-arm: population match at the selected checkpoint ===")
    print("metric_mmd and metric_w2 are logged in every arm, so this is the one")
    print("table on which objectives can be compared -- and unlike loss_pair it is")
    print("not capped by MOSCOT pairing noise.\n")
    header = (
        f"{'#':>2} {'config':<24}{'arm':>10}{'mmd':>11}{'w2':>11}{'loss_pair':>11}{'ep':>4}"
    )
    print(header)
    print("-" * len(header))
    for i, r in enumerate(scored, 1):
        w2 = "  n/a" if np.isnan(r["w2"]) else f"{r['w2']:.4f}"
        lp = "  n/a" if np.isnan(r["loss_pair"]) else f"{r['loss_pair']:.6f}"
        print(
            f"{i:>2} {r['tag']:<24}{r['arm']:>10}{r['mmd']:>11.5f}{w2:>11}{lp:>11}{r['epoch']:>4}"
        )

    floors = {r["arm"]: r["mmd"] for r in scored if is_level_zero(r)}
    for arm, mmd in sorted(floors.items()):
        print(f"\n  identity floor ({arm}): mmd = {mmd:.5f}")
    if floors:
        best_floor = min(floors.values())
        beat = [r for r in scored if r["mmd"] < best_floor and not is_level_zero(r)]
        print(f"  {len(beat)} configuration(s) match the post-treatment distribution")
        print("  better than leaving the cells untouched.")
        if not beat:
            print("  -> None do. Every model made the population match WORSE than identity")
            print("     while improving per-cell MSE, which is regression to the mean.")


def main() -> None:
    args = parse_args()
    grouped, skipped, directories = collect(args.pattern, args.metric)

    if skipped:
        print(f"skipped {len(skipped)} directory(ies):")
        for directory, why in skipped[:8]:
            print(f"  {os.path.basename(directory)}: {why}")
        print()

    if not grouped:
        raise SystemExit("nothing to rank")

    rows = aggregate(grouped)
    print(f"configs: {len(rows)}   directories: {len(directories)}")

    by_arm: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_arm[row["arm"]].append(row)

    promoted: dict[str, list[dict]] = {}
    for arm in sorted(by_arm):
        promoted[arm] = print_arm(arm, by_arm[arm], args.top)

    print_cross_arm(rows)

    if any(r["n_folds"] == 1 for r in rows):
        print("\nNOTE: single-fold results. Small differences are not separable from fold")
        print("      noise -- use this to pick candidates, not a winner.")

    print("\nSuggested promotion to full cross-validation:")
    any_promoted = False
    for arm in sorted(promoted):
        picks = promoted[arm]
        if not picks:
            print(f"  {arm}: nothing beat the floor -- promote nothing from this arm.")
            continue
        any_promoted = True
        for r in picks:
            print(
                f"  [{arm}] {r['tag']:<24} val_{r['metric']}={r['best']:.6f}  "
                f"(level {r['level']}, {r['variant']})"
            )
    if not any_promoted:
        print("  Nothing cleared its floor in either arm. The next step is diagnostic,")
        print("  not a CV run.")


if __name__ == "__main__":
    main()
