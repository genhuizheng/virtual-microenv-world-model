#!/usr/bin/env python3
"""Rank Stage-1 VAE configurations from evaluate_stage1_scvi.py outputs.

There is no single correct ranking here, and pretending otherwise would hide
the actual decision. Stage 1 is judged on three things that trade against each
other:

    reconstruction   can the latent rebuild the cell?      (higher Pearson better)
    batch mixing     is the latent organised by study?     (ratio near 1 better)
    role signal      is pre/post treatment still visible?  (ratio well above 1 better)

Measured on this project's data, mixing and role move together: configurations
that mix studies harder also retain less treatment signal. So "best mixing"
and "best role" are usually different configurations, and a ranking that
optimises either alone is misleading.

This script therefore prints per-metric ranks side by side, then a weighted
composite whose weights are explicit and adjustable. The default weighting
favours role signal, because a representation that has erased the pre/post
distinction cannot support a transition model no matter how well it
reconstructs -- the ceiling for Stage 2 is set here. Reconstruction is second
(the decoder has to produce usable expression for downstream DEG), and mixing
third, because the MOSCOT pairs are within-patient: the transition model never
compares cells across studies, so residual study structure costs less than the
raw ratio suggests.

Re-weight with --weights if your priorities differ; the per-metric columns are
always shown so the composite never hides the underlying trade.

Usage:
    python rank_stage1_screen.py --pattern 'eval_screen_*_fold0.json'
    python rank_stage1_screen.py --pattern 'eval_*.json' --weights role=1,recon=1,mix=1
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from collections import defaultdict

import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--pattern", default="eval_screen_*.json")
    p.add_argument(
        "--weights",
        default="role=0.5,recon=0.3,mix=0.2",
        help="Composite weights as key=value pairs over {role,recon,mix}. "
        "Defaults favour role signal; see the module docstring for why.",
    )
    p.add_argument("--top", type=int, default=3, help="How many to recommend promoting.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    weights = {"role": 0.5, "recon": 0.3, "mix": 0.2}
    for item in args.weights.split(","):
        if not item.strip():
            continue
        k, _, v = item.partition("=")
        k = k.strip()
        if k not in weights:
            raise SystemExit(f"unknown weight {k!r}; expected one of {sorted(weights)}")
        weights[k] = float(v)
    total_w = sum(weights.values())
    if total_w <= 0:
        raise SystemExit("weights must sum to a positive value")

    files = sorted(glob.glob(args.pattern))
    if not files:
        raise SystemExit(f"no files matched {args.pattern!r}")

    # Group by config so multi-fold results average rather than compete.
    grouped = defaultdict(list)
    for path in files:
        d = json.load(open(path))
        # Label by the checkpoint's directory name, which is the config tag.
        # Handles absolute paths and either separator.
        tag = os.path.basename(os.path.dirname(d["checkpoint"].replace("\\", "/")))
        for prefix in ("stage1_screen_", "stage1_"):
            if tag.startswith(prefix):
                tag = tag[len(prefix):]
                break
        grouped[tag].append(d)

    rows = []
    for tag, records in grouped.items():
        pearson = np.array([r["recon_pearson_mean"] for r in records], dtype=float)
        # MSE is in the units decode() returns, which differ across input
        # representations (log1p vs bin indices). It is reported for reference
        # but never used in the composite; Pearson is the scale-invariant
        # reconstruction metric that survives the comparison.
        mse = np.array(
            [r.get("recon_mse", r.get("recon_mse_log1p")) or np.nan for r in records], dtype=float
        )
        mix = np.array(
            [r["knn_same_source_study"] / max(r["knn_source_study_chance"], 1e-12) for r in records],
            dtype=float,
        )
        role = np.array(
            [r["knn_same_role"] / max(r["knn_role_chance"], 1e-12) for r in records], dtype=float
        )
        rows.append(
            {
                "tag": tag,
                "n_folds": len(records),
                "likelihood": records[0].get("gene_likelihood", "?"),
                "transform": records[0].get("expression_transform", "?"),
                "n_cells": records[0].get("n_cells", 0),
                "units": records[0].get("recon_units", "?"),
                "pearson": float(np.nanmean(pearson)),
                "pearson_sd": float(np.nanstd(pearson)),
                "mse": float(np.nanmean(mse)),
                "mix": float(np.nanmean(mix)),
                "role": float(np.nanmean(role)),
            }
        )

    if len(rows) < 2:
        print("WARNING: only one configuration found; ranking is meaningless.\n")

    # Normalize each metric to [0, 1] where 1 is best, so the composite is not
    # dominated by whichever metric happens to have the largest raw range.
    def scale(values: np.ndarray, higher_is_better: bool) -> np.ndarray:
        lo, hi = float(np.nanmin(values)), float(np.nanmax(values))
        if not np.isfinite(lo) or hi - lo < 1e-12:
            return np.full_like(values, 0.5)
        unit = (values - lo) / (hi - lo)
        return unit if higher_is_better else 1.0 - unit

    pearson = np.array([r["pearson"] for r in rows])
    mixv = np.array([r["mix"] for r in rows])
    rolev = np.array([r["role"] for r in rows])

    s_recon = scale(pearson, True)
    s_mix = scale(mixv, False)   # lower ratio = better mixing
    s_role = scale(rolev, True)

    for i, r in enumerate(rows):
        r["score"] = float(
            (weights["role"] * s_role[i] + weights["recon"] * s_recon[i] + weights["mix"] * s_mix[i])
            / total_w
        )

    def rank_of(values: np.ndarray, higher_is_better: bool) -> dict[str, int]:
        order = np.argsort(-values if higher_is_better else values)
        return {rows[idx]["tag"]: pos + 1 for pos, idx in enumerate(order)}

    r_recon = rank_of(pearson, True)
    r_mix = rank_of(mixv, False)
    r_role = rank_of(rolev, True)

    rows.sort(key=lambda r: -r["score"])

    print(f"weights: role={weights['role']} recon={weights['recon']} mix={weights['mix']}")
    print(f"configs: {len(rows)}   files: {len(files)}\n")
    header = (
        f"{'#':>2} {'config':<20}{'lik':>8}{'fold':>5}"
        f"{'Pearson':>10}{'MSE':>9}{'units':>11}{'mix':>8}{'role':>8}"
        f"{'rP':>4}{'rM':>4}{'rR':>4}{'score':>8}"
    )
    print(header)
    print("-" * len(header))
    for i, r in enumerate(rows, 1):
        print(
            f"{i:>2} {r['tag']:<20}{r['likelihood']:>8}{r['n_folds']:>5}"
            f"{r['pearson']:>10.4f}{r['mse']:>9.3f}{r['units']:>11}{r['mix']:>7.2f}x{r['role']:>7.2f}x"
            f"{r_recon[r['tag']]:>4}{r_mix[r['tag']]:>4}{r_role[r['tag']]:>4}{r['score']:>8.3f}"
        )
    print("\nrP/rM/rR = rank by Pearson / mixing / role  (1 = best)")
    print("mix: lower is better (1.0 = fully mixed).  role: higher is better (1.0 = erased).")
    if len({r["units"] for r in rows}) > 1:
        print()
        print("NOTE: configurations span different reconstruction units, so the MSE")
        print("      column is NOT comparable across rows. Pearson, mixing and role")
        print("      are; the composite uses Pearson and ignores MSE for that reason.")
    if any(r["n_folds"] == 1 for r in rows):
        print(
            "\nNOTE: some configurations have a single fold. Differences of a few percent "
            "are not separable from fold noise -- use this to pick candidates, not a winner."
        )

    top = rows[: max(args.top, 1)]
    print(f"\nSuggested promotion to full cross-validation ({len(top)}):")
    for r in top:
        print(f"  {r['tag']:<20} score={r['score']:.3f}  "
              f"(Pearson {r['pearson']:.4f}, mix {r['mix']:.2f}x, role {r['role']:.2f}x)")

    # Surface the tension explicitly rather than letting the composite bury it.
    best_role = max(rows, key=lambda r: r["role"])
    best_mix = min(rows, key=lambda r: r["mix"])
    best_recon = max(rows, key=lambda r: r["pearson"])
    print("\nPer-metric winners (these are usually different configurations):")
    print(f"  best role         : {best_role['tag']} ({best_role['role']:.2f}x)")
    print(f"  best batch mixing : {best_mix['tag']} ({best_mix['mix']:.2f}x)")
    print(f"  best reconstruction: {best_recon['tag']} (Pearson {best_recon['pearson']:.4f})")
    if len({best_role["tag"], best_mix["tag"], best_recon["tag"]}) > 1:
        print("  -> no configuration dominates; the composite weighting is a real choice.")


if __name__ == "__main__":
    main()
