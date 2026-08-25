#!/usr/bin/env python3
"""Screen high-confidence OT biological outputs for breast, prostate, and global results."""

from __future__ import annotations

import argparse
import csv
import math
import re
from pathlib import Path


DEFAULT_OUT_DIR = Path("batch_processed_unified_genes_zero_fill") / "initial_biological_analysis_global"

BREAST_PATTERNS = ("breast", "mammary")
PROSTATE_PATTERNS = ("prostate", "prostatic", "prostrate")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument("--metric", type=str, default=None, help="Optional confidence metric filter.")
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise SystemExit(f"Missing file: {path}")
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def as_float(value: str) -> float:
    if value in ("", "nan", "NaN", "None", None):
        return math.nan
    return float(value)


def matches_any(text: str, patterns: tuple[str, ...]) -> bool:
    low = str(text).lower()
    return any(pattern in low for pattern in patterns)


def category(row: dict[str, str]) -> str | None:
    scope = row.get("scope", "")
    tumor = row.get("tumor_type", "")
    if scope == "global" or str(tumor).lower() == "all":
        return "global"
    if matches_any(tumor, BREAST_PATTERNS):
        return "breast"
    if matches_any(tumor, PROSTATE_PATTERNS):
        return "prostate"
    return None


def print_csv_table(rows: list[dict[str, str]], fields: list[str]) -> None:
    print(",".join(fields))
    for row in rows:
        print(",".join(str(row.get(field, "")) for field in fields))


def summarize_de_outputs(out_dir: Path, metric_filter: str | None) -> list[dict[str, str]]:
    rows = read_csv(out_dir / "formal_de_outputs_by_tumor_metric.csv")
    selected = []
    for row in rows:
        cat = category(row)
        if cat is None:
            continue
        if metric_filter and row.get("metric") != metric_filter:
            continue
        selected.append(
            {
                "category": cat,
                "scope": row.get("scope", ""),
                "tumor_type": row.get("tumor_type", ""),
                "metric": row.get("metric", ""),
                "n_pairs": row.get("n_pairs", ""),
                "n_replicates": row.get("n_replicates", ""),
                "n_fdr_0p05": row.get("n_fdr_0p05", ""),
                "n_fdr_0p10": row.get("n_fdr_0p10", ""),
                "top_gene_csv": row.get("top_gene_csv", ""),
            }
        )
    selected.sort(
        key=lambda row: (
            {"global": 0, "breast": 1, "prostate": 2}.get(row["category"], 9),
            row["metric"],
            row["tumor_type"],
        )
    )
    return selected


def top_genes_for_summary_row(row: dict[str, str], top_n: int) -> list[dict[str, str]]:
    path = Path(row["top_gene_csv"])
    if not path.exists():
        return []
    genes = read_csv(path)
    # Prefer significant genes; if none, use the top rows already written by the analysis script.
    significant = [g for g in genes if as_float(g.get("paired_t_fdr_bh", "nan")) <= 0.10]
    chosen = significant[:top_n] if significant else genes[:top_n]
    out = []
    for gene in chosen:
        out.append(
            {
                "category": row["category"],
                "tumor_type": row["tumor_type"],
                "metric": row["metric"],
                "gene": gene.get("gene_symbol") or gene.get("gene_id") or gene.get("gene", ""),
                "mean_delta": gene.get("mean_delta", ""),
                "fdr": gene.get("paired_t_fdr_bh", ""),
                "pvalue": gene.get("paired_t_pvalue", ""),
            }
        )
    return out


def safe_name(text: str) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text)
    return text.strip("_") or "unknown"


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir
    summary = summarize_de_outputs(out_dir, args.metric)

    print("=== BREAST / PROSTATE / GLOBAL DE SUMMARY ===")
    print_csv_table(
        summary,
        [
            "category",
            "scope",
            "tumor_type",
            "metric",
            "n_pairs",
            "n_replicates",
            "n_fdr_0p05",
            "n_fdr_0p10",
            "top_gene_csv",
        ],
    )

    print()
    print(f"=== TOP GENES PER RESULT, top_n={args.top_n} ===")
    top_rows = []
    for row in summary:
        top_rows.extend(top_genes_for_summary_row(row, args.top_n))
    print_csv_table(top_rows, ["category", "tumor_type", "metric", "gene", "mean_delta", "fdr", "pvalue"])

    out_csv = out_dir / f"screen_breast_prostate_global_summary_{safe_name(args.metric or 'all_metrics')}.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "category",
                "scope",
                "tumor_type",
                "metric",
                "n_pairs",
                "n_replicates",
                "n_fdr_0p05",
                "n_fdr_0p10",
                "top_gene_csv",
            ],
        )
        writer.writeheader()
        writer.writerows(summary)
    print()
    print(f"wrote: {out_csv}")


if __name__ == "__main__":
    main()
