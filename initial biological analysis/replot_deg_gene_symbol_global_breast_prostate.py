#!/usr/bin/env python3
"""Replot DEG figures with gene symbols for global, breast, and prostate results."""

from __future__ import annotations

import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# ======================
# Local variables
# ======================
RESULT_DIR = Path(r"C:\Users\ncsub\Downloads\initial_biological_analysis_global")
EXPORT_DIR = RESULT_DIR / "screen_breast_prostate_global_export" / "plots_gene_symbol"

# Use None for all confidence metrics, or set e.g. "combined_concentration"
METRIC = None

TOP_N = 30
LABEL_N = 12
FDR_CUTOFF = 0.05
OVERLAP_TOP_N = 200

BREAST_KEYS = ("breast", "mammary")
PROSTATE_KEYS = ("prostate", "prostatic", "prostrate")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def safe_text(x: object) -> str:
    return "".join(c if c.isalnum() or c in "._-" else "_" for c in str(x)).strip("_")[:100] or "unknown"


def fnum(x: object) -> float:
    try:
        if x in ("", "nan", "NaN", "None", None):
            return math.nan
        return float(x)
    except Exception:
        return math.nan


def match_category(scope: str, tumor_type: str) -> str | None:
    text = str(tumor_type).lower()
    if str(scope).lower() == "global" or text == "all":
        return "global"
    if any(k in text for k in BREAST_KEYS):
        return "breast"
    if any(k in text for k in PROSTATE_KEYS):
        return "prostate"
    return None


def localize_path(path_text: str, result_dir: Path) -> Path | None:
    p = Path(path_text)
    if p.exists():
        return p
    p2 = result_dir / p.name
    if p2.exists():
        return p2
    return None


def gene_label(row: dict[str, str]) -> str:
    # Prefer HGNC symbols. Some existing plots used gene_name, which can be ENSG.
    for key in ("gene_symbol", "gene", "gene_name"):
        value = str(row.get(key, "")).strip()
        if value and not value.upper().startswith("ENSG"):
            return value
    for key in ("gene_id", "ensembl_gene_id", "index"):
        value = str(row.get(key, "")).strip()
        if value:
            return value
    return ""


def delta_value(row: dict[str, str]) -> float:
    for key in ("pseudobulk_paired_delta", "mean_delta_target_minus_source", "mean_delta"):
        value = fnum(row.get(key))
        if math.isfinite(value):
            return value
    return math.nan


def annotate_rows(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    out = []
    for row in rows:
        delta = delta_value(row)
        pvalue = fnum(row.get("paired_t_pvalue"))
        fdr = fnum(row.get("paired_t_fdr_bh"))
        out.append(
            {
                **row,
                "_gene_label": gene_label(row),
                "_delta": delta,
                "_abs_delta": abs(delta) if math.isfinite(delta) else math.nan,
                "_pvalue": pvalue,
                "_fdr": fdr,
                "_neglog10_fdr": -math.log10(max(fdr, 1e-300)) if math.isfinite(fdr) and fdr > 0 else 0.0,
                "_neglog10_p": -math.log10(max(pvalue, 1e-300)) if math.isfinite(pvalue) and pvalue > 0 else 0.0,
            }
        )
    return out


def plot_top_gene_deltas(rows: list[dict[str, object]], out_path: Path, title: str) -> None:
    top = sorted(
        [r for r in rows if math.isfinite(float(r["_abs_delta"]))],
        key=lambda r: float(r["_abs_delta"]),
        reverse=True,
    )[:TOP_N]
    top = list(reversed(top))
    if not top:
        return

    labels = [str(r["_gene_label"]) for r in top]
    values = [float(r["_delta"]) for r in top]
    colors = ["#d95f02" if v > 0 else "#1f78b4" for v in values]

    fig, ax = plt.subplots(figsize=(8.5, max(5.0, 0.30 * len(top))))
    ax.barh(range(len(top)), values, color=colors, alpha=0.88)
    ax.axvline(0, color="black", linewidth=1.0)
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Post - pre pseudobulk expression delta", fontweight="bold")
    ax.set_title(title, fontsize=11)
    ax.grid(axis="x", alpha=0.18)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_volcano(rows: list[dict[str, object]], out_path: Path, title: str) -> None:
    valid = [
        r
        for r in rows
        if math.isfinite(float(r["_delta"])) and math.isfinite(float(r["_neglog10_fdr"]))
    ]
    if not valid:
        return

    x = np.array([float(r["_delta"]) for r in valid])
    y = np.array([float(r["_neglog10_fdr"]) for r in valid])
    sig = np.array([math.isfinite(float(r["_fdr"])) and float(r["_fdr"]) <= FDR_CUTOFF for r in valid])

    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    ax.scatter(x[~sig], y[~sig], s=8, c="lightgray", alpha=0.65, linewidths=0)
    ax.scatter(x[sig], y[sig], s=16, c="#d73027", alpha=0.85, linewidths=0)
    ax.axvline(0, color="black", linestyle="--", linewidth=1.0)
    ax.axhline(-math.log10(FDR_CUTOFF), color="black", linestyle=":", linewidth=1.0)
    ax.set_xlabel("Post - pre pseudobulk expression delta", fontweight="bold")
    ax.set_ylabel("-log10(FDR)", fontweight="bold")
    ax.set_title(title, fontsize=11)
    ax.grid(alpha=0.15)

    label_rows = sorted(
        valid,
        key=lambda r: (
            float(r["_fdr"]) if math.isfinite(float(r["_fdr"])) else math.inf,
            -float(r["_abs_delta"]) if math.isfinite(float(r["_abs_delta"])) else 0.0,
        ),
    )[:LABEL_N]
    for row in label_rows:
        label = str(row["_gene_label"])
        if not label:
            continue
        ax.text(
            float(row["_delta"]),
            float(row["_neglog10_fdr"]),
            label,
            fontsize=7,
            ha="left",
            va="bottom",
        )

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def gene_sets_for_overlap(rows: list[dict[str, object]]) -> dict[str, set[str]]:
    valid = [r for r in rows if str(r["_gene_label"]) and not str(r["_gene_label"]).upper().startswith("ENSG")]
    fdr_genes = {
        str(r["_gene_label"])
        for r in valid
        if math.isfinite(float(r["_fdr"])) and float(r["_fdr"]) <= FDR_CUTOFF
    }
    top_delta_genes = {
        str(r["_gene_label"])
        for r in sorted(
            [r for r in valid if math.isfinite(float(r["_abs_delta"]))],
            key=lambda r: float(r["_abs_delta"]),
            reverse=True,
        )[:OVERLAP_TOP_N]
    }
    top_pvalue_genes = {
        str(r["_gene_label"])
        for r in sorted(
            [r for r in valid if math.isfinite(float(r["_pvalue"]))],
            key=lambda r: float(r["_pvalue"]),
        )[:OVERLAP_TOP_N]
    }
    return {
        "fdr_0p05": fdr_genes,
        f"top{OVERLAP_TOP_N}_abs_delta": top_delta_genes,
        f"top{OVERLAP_TOP_N}_pvalue": top_pvalue_genes,
    }


def jaccard(a: set[str], b: set[str]) -> float:
    union = a | b
    if not union:
        return math.nan
    return len(a & b) / len(union)


def overlap_rows_for_group(category: str, tumor_type: str, metric_sets: dict[str, dict[str, set[str]]]) -> list[dict[str, object]]:
    rows = []
    metrics = sorted(metric_sets)
    set_names = sorted({name for sets in metric_sets.values() for name in sets})

    for set_name in set_names:
        available_sets = [metric_sets[m].get(set_name, set()) for m in metrics]
        nonempty_sets = [s for s in available_sets if s]
        shared_all = set.intersection(*nonempty_sets) if nonempty_sets else set()
        union_all = set.union(*nonempty_sets) if nonempty_sets else set()
        rows.append(
            {
                "row_type": "summary",
                "category": category,
                "tumor_type": tumor_type,
                "gene_set": set_name,
                "metric_1": "ALL_METRICS",
                "metric_2": "",
                "n_metric_1": "",
                "n_metric_2": "",
                "n_overlap": len(shared_all),
                "n_union": len(union_all),
                "jaccard": len(shared_all) / len(union_all) if union_all else math.nan,
                "overlap_genes": ";".join(sorted(shared_all)),
            }
        )

        for i, m1 in enumerate(metrics):
            for m2 in metrics[i + 1:]:
                s1 = metric_sets[m1].get(set_name, set())
                s2 = metric_sets[m2].get(set_name, set())
                overlap = s1 & s2
                union = s1 | s2
                rows.append(
                    {
                        "row_type": "pairwise",
                        "category": category,
                        "tumor_type": tumor_type,
                        "gene_set": set_name,
                        "metric_1": m1,
                        "metric_2": m2,
                        "n_metric_1": len(s1),
                        "n_metric_2": len(s2),
                        "n_overlap": len(overlap),
                        "n_union": len(union),
                        "jaccard": jaccard(s1, s2),
                        "overlap_genes": ";".join(sorted(overlap)),
                    }
                )
    return rows


def plot_overlap_heatmaps(overlap_rows: list[dict[str, object]], out_dir: Path) -> list[dict[str, str]]:
    plot_rows = []
    pairwise = [r for r in overlap_rows if r["row_type"] == "pairwise"]
    grouped: dict[tuple[str, str, str], list[dict[str, object]]] = {}
    for row in pairwise:
        key = (str(row["category"]), str(row["tumor_type"]), str(row["gene_set"]))
        grouped.setdefault(key, []).append(row)

    for (category, tumor_type, gene_set), rows in grouped.items():
        metrics = sorted({str(r["metric_1"]) for r in rows} | {str(r["metric_2"]) for r in rows})
        if len(metrics) < 2:
            continue
        idx = {m: i for i, m in enumerate(metrics)}
        mat = np.full((len(metrics), len(metrics)), np.nan, dtype=float)
        text_mat = [["" for _ in metrics] for _ in metrics]

        # Fill diagonal only when the metric has a non-empty gene set.
        counts = {m: 0 for m in metrics}
        for row in rows:
            m1 = str(row["metric_1"])
            m2 = str(row["metric_2"])
            counts[m1] = max(counts[m1], int(row["n_metric_1"]))
            counts[m2] = max(counts[m2], int(row["n_metric_2"]))
        for metric, count in counts.items():
            i = idx[metric]
            if count > 0:
                mat[i, i] = 1.0
                text_mat[i][i] = f"n={count}"
            else:
                text_mat[i][i] = "n=0"

        for row in rows:
            i = idx[str(row["metric_1"])]
            j = idx[str(row["metric_2"])]
            value = float(row["jaccard"]) if math.isfinite(float(row["jaccard"])) else np.nan
            mat[i, j] = value
            mat[j, i] = value
            overlap = int(row["n_overlap"])
            union = int(row["n_union"])
            if union > 0 and math.isfinite(value):
                text = f"{value:.2f}\n{overlap}/{union}"
            else:
                text = "0/0"
            text_mat[i][j] = text
            text_mat[j][i] = text

        fig, ax = plt.subplots(figsize=(max(6.5, 0.65 * len(metrics)), max(5.8, 0.58 * len(metrics))))
        cmap = plt.colormaps.get_cmap("viridis").copy()
        cmap.set_bad(color="#f2f2f2")
        im = ax.imshow(mat, cmap=cmap, vmin=0, vmax=1)
        ax.set_xticks(range(len(metrics)))
        ax.set_yticks(range(len(metrics)))
        ax.set_xticklabels(metrics, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(metrics, fontsize=8)
        ax.set_title(f"Gene overlap consistency: {tumor_type}\n{gene_set}", fontsize=11)
        for i in range(len(metrics)):
            for j in range(len(metrics)):
                text = text_mat[i][j]
                if not text:
                    continue
                color = "black"
                if math.isfinite(mat[i, j]) and mat[i, j] > 0.55:
                    color = "white"
                ax.text(j, i, text, ha="center", va="center", fontsize=6.5, color=color)
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Jaccard overlap")
        fig.tight_layout()
        plot_path = out_dir / "overlap_heatmaps" / category / f"gene_overlap_jaccard_{safe_text(category)}_{safe_text(tumor_type)}_{safe_text(gene_set)}.png"
        plot_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        plot_rows.append(
            {
                "category": category,
                "tumor_type": tumor_type,
                "gene_set": gene_set,
                "plot": str(plot_path),
            }
        )
    return plot_rows


def main() -> None:
    summary_path = RESULT_DIR / "formal_de_outputs_by_tumor_metric.csv"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing: {summary_path}")

    plot_index = []
    metric_gene_sets: dict[tuple[str, str], dict[str, dict[str, set[str]]]] = {}
    for summary in read_csv(summary_path):
        scope = summary.get("scope", "")
        tumor_type = summary.get("tumor_type", "")
        metric = summary.get("metric", "")
        if METRIC is not None and metric != METRIC:
            continue

        category = match_category(scope, tumor_type)
        if category is None:
            continue

        gene_csv = localize_path(summary.get("gene_csv", ""), RESULT_DIR)
        if gene_csv is None:
            continue

        rows = annotate_rows(read_csv(gene_csv))
        metric_gene_sets.setdefault((category, tumor_type), {})[metric] = gene_sets_for_overlap(rows)
        safe_metric = safe_text(metric)
        safe_tumor = safe_text(tumor_type)
        out_dir = EXPORT_DIR / category / safe_metric
        title_suffix = f"{tumor_type} / {metric}"

        top_path = out_dir / f"top_gene_deltas_gene_symbol_{safe_text(category)}_{safe_tumor}_{safe_metric}.png"
        volcano_path = out_dir / f"volcano_gene_symbol_{safe_text(category)}_{safe_tumor}_{safe_metric}.png"

        plot_top_gene_deltas(rows, top_path, f"Top gene changes: {title_suffix}")
        plot_volcano(rows, volcano_path, f"Paired pseudobulk DE: {title_suffix}")

        plot_index.append(
            {
                "category": category,
                "tumor_type": tumor_type,
                "metric": metric,
                "gene_csv": str(gene_csv),
                "top_gene_delta_plot": str(top_path),
                "volcano_plot": str(volcano_path),
            }
        )
        print(f"done: {category} / {tumor_type} / {metric}")

    index_path = EXPORT_DIR / "gene_symbol_replot_index.csv"
    index_path.parent.mkdir(parents=True, exist_ok=True)
    with index_path.open("w", newline="", encoding="utf-8") as f:
        fields = ["category", "tumor_type", "metric", "gene_csv", "top_gene_delta_plot", "volcano_plot"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(plot_index)

    overlap_rows = []
    for (category, tumor_type), metric_sets in sorted(metric_gene_sets.items()):
        overlap_rows.extend(overlap_rows_for_group(category, tumor_type, metric_sets))

    overlap_path = EXPORT_DIR / "gene_symbol_metric_overlap_summary.csv"
    with overlap_path.open("w", newline="", encoding="utf-8") as f:
        fields = [
            "row_type",
            "category",
            "tumor_type",
            "gene_set",
            "metric_1",
            "metric_2",
            "n_metric_1",
            "n_metric_2",
            "n_overlap",
            "n_union",
            "jaccard",
            "overlap_genes",
        ]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(overlap_rows)

    overlap_plot_rows = plot_overlap_heatmaps(overlap_rows, EXPORT_DIR)
    overlap_plot_path = EXPORT_DIR / "gene_symbol_metric_overlap_plots.csv"
    with overlap_plot_path.open("w", newline="", encoding="utf-8") as f:
        fields = ["category", "tumor_type", "gene_set", "plot"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(overlap_plot_rows)

    print("Wrote:")
    print(index_path)
    print(overlap_path)
    print(overlap_plot_path)
    print(EXPORT_DIR)


if __name__ == "__main__":
    main()
