#!/usr/bin/env python3
"""Concatenate DEG Enrichr results and make global-only enrichment plots."""

from __future__ import annotations

import csv
import math
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# ======================
# Local variables
# ======================
RESULT_DIR = Path(r"C:\Users\ncsub\Downloads\initial_biological_analysis_global")
ENRICHMENT_DIR = RESULT_DIR / "screen_breast_prostate_global_deg_enrichment"
EXPORT_DIR = ENRICHMENT_DIR / "global_only_enrichr_plots"

# Use None for all metrics, or set e.g. "combined_concentration"
METRIC = None

FDR_PLOT_CUTOFF = 0.25
PVALUE_PLOT_CUTOFF = 0.05
PLOT_SCORE = "pvalue"  # "pvalue" is useful when Enrichr FDR is too conservative; table still reports FDR.
TOP_TERMS_PER_CATEGORY = 10
MAX_TERMS_PER_PLOT = 20

CATEGORY_ORDER = ["global"]
CATEGORY_LABELS = {"global": "Global", "breast": "Breast", "prostate": "Prostate"}
CATEGORY_X = {"global": 0.0}
DIRECTION_ORDER = ["up_after_treatment", "down_after_treatment", "all_degs"]
LIBRARY_MARKERS = {"GO BP": "o", "KEGG": "s", "Reactome": "^"}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def safe_text(x: object) -> str:
    return "".join(c if c.isalnum() or c in "._-" else "_" for c in str(x)).strip("_")[:100] or "unknown"


def fnum(x: object) -> float:
    if x in ("", None, "nan", "NaN", "None"):
        return math.nan
    try:
        return float(x)
    except Exception:
        return math.nan


def adjusted_p_value(row: dict[str, str]) -> float:
    for key in ("Adjusted P-value", "Adjusted P Value", "Adjusted_P_value", "FDR", "fdr"):
        value = fnum(row.get(key))
        if math.isfinite(value):
            return value
    return math.nan


def raw_p_value(row: dict[str, str]) -> float:
    for key in ("P-value", "P Value", "P_value", "pvalue", "p_value"):
        value = fnum(row.get(key))
        if math.isfinite(value):
            return value
    return math.nan


def combined_score(row: dict[str, str]) -> float:
    for key in ("Combined Score", "Combined_Score", "combined_score"):
        value = fnum(row.get(key))
        if math.isfinite(value):
            return value
    return math.nan


def odds_ratio(row: dict[str, str]) -> float:
    for key in ("Odds Ratio", "Odds_Ratio", "odds_ratio"):
        value = fnum(row.get(key))
        if math.isfinite(value):
            return value
    return math.nan


def overlap_count(overlap: str) -> int:
    try:
        return int(str(overlap).split("/", 1)[0])
    except Exception:
        return 0


def overlap_total(overlap: str) -> int:
    try:
        return int(str(overlap).split("/", 1)[1])
    except Exception:
        return 0


def pretty_library(library: str) -> str:
    if library.startswith("GO_"):
        return "GO BP"
    if library.startswith("KEGG"):
        return "KEGG"
    if library.startswith("Reactome"):
        return "Reactome"
    return library


def clean_term(term: str) -> str:
    term = str(term)
    if "(GO:" in term:
        term = term.split("(GO:", 1)[0].strip()
    return term[:100]


def wrap_label(label: str, width: int = 46) -> str:
    return "\n".join(textwrap.wrap(str(label), width=width, break_long_words=False))


def parse_result_path(path: Path) -> dict[str, str] | None:
    # Expected:
    # enrichment_results/<label>/<direction>/<library>/enrichr_results.csv
    parts = path.parts
    try:
        idx = parts.index("enrichment_results")
    except ValueError:
        return None
    if len(parts) <= idx + 3:
        return None

    label = parts[idx + 1]
    direction = parts[idx + 2]
    library = parts[idx + 3]

    category = ""
    tumor_type = ""
    metric = ""
    for prefix in CATEGORY_ORDER:
        if label.startswith(prefix + "_"):
            category = prefix
            rest = label[len(prefix) + 1 :]
            known_metrics = [
                "combined_concentration",
                "effective_target_confidence",
                "entropy_confidence",
                "max_coupling",
                "retained_mass",
                "top12_margin",
                "top12_ratio_log",
            ]
            for m in known_metrics:
                suffix = "_" + m
                if rest.endswith(suffix):
                    tumor_type = rest[: -len(suffix)]
                    metric = m
                    break
            break

    if not category or not metric:
        return None

    return {
        "category": category,
        "tumor_type": tumor_type,
        "metric": metric,
        "direction": direction,
        "library": library,
    }


def load_all_results() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for path in ENRICHMENT_DIR.glob("enrichment_results/**/enrichr_results.csv"):
        meta = parse_result_path(path)
        if meta is None:
            continue
        if METRIC is not None and meta["metric"] != METRIC:
            continue
        for row in read_csv(path):
            fdr = adjusted_p_value(row)
            if not math.isfinite(fdr) or fdr <= 0:
                continue
            pvalue = raw_p_value(row)
            term = row.get("Term", "")
            if not term:
                continue
            overlap = row.get("Overlap", "")
            library_pretty = pretty_library(meta["library"])
            rows.append(
                {
                    **meta,
                    "term": clean_term(term),
                    "term_with_library": f"{clean_term(term)} [{library_pretty}]",
                    "library_pretty": library_pretty,
                    "pvalue": pvalue,
                    "fdr": fdr,
                    "neglog10_pvalue": -math.log10(max(pvalue, 1e-300)) if math.isfinite(pvalue) and pvalue > 0 else math.nan,
                    "neglog10_fdr": -math.log10(max(fdr, 1e-300)),
                    "overlap": overlap,
                    "overlap_count": overlap_count(overlap),
                    "overlap_total": overlap_total(overlap),
                    "odds_ratio": odds_ratio(row),
                    "combined_score": combined_score(row),
                    "genes": row.get("Genes", ""),
                    "source_file": str(path),
                }
            )
    return rows


def write_combined_csv(rows: list[dict[str, object]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "category",
        "tumor_type",
        "metric",
        "direction",
        "library",
        "library_pretty",
        "term",
        "term_with_library",
        "pvalue",
        "fdr",
        "neglog10_pvalue",
        "neglog10_fdr",
        "overlap",
        "overlap_count",
        "overlap_total",
        "odds_ratio",
        "combined_score",
        "genes",
        "source_file",
    ]
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def choose_terms(rows: list[dict[str, object]]) -> list[str]:
    candidate_terms = []
    for category in CATEGORY_ORDER:
        subset = [r for r in rows if r["category"] == category and include_for_plot(r)]
        subset = sorted(subset, key=lambda r: (-plot_score(r), -int(r.get("overlap_count", 0))))
        candidate_terms.extend(str(r["term_with_library"]) for r in subset[:TOP_TERMS_PER_CATEGORY])

    # Stable unique order, strongest terms first.
    best_score: dict[str, float] = {}
    for row in rows:
        term = str(row["term_with_library"])
        score = plot_score(row)
        best_score[term] = max(best_score.get(term, 0.0), score)
    unique_terms = sorted(set(candidate_terms), key=lambda t: best_score.get(t, 0.0), reverse=True)
    return unique_terms[:MAX_TERMS_PER_PLOT]


def plot_score(row: dict[str, object]) -> float:
    if PLOT_SCORE == "fdr":
        value = row.get("neglog10_fdr", math.nan)
    else:
        value = row.get("neglog10_pvalue", math.nan)
    value = float(value) if value not in ("", None) else math.nan
    return value if math.isfinite(value) else 0.0


def include_for_plot(row: dict[str, object]) -> bool:
    pvalue = float(row.get("pvalue", math.nan))
    fdr = float(row.get("fdr", math.nan))
    if PLOT_SCORE == "fdr":
        return math.isfinite(fdr) and fdr <= FDR_PLOT_CUTOFF
    return math.isfinite(pvalue) and pvalue <= PVALUE_PLOT_CUTOFF


def plot_metric_direction(rows: list[dict[str, object]], metric: str, direction: str, out_dir: Path) -> str | None:
    subset = [r for r in rows if r["metric"] == metric and r["direction"] == direction and include_for_plot(r)]
    if not subset:
        return None

    terms = choose_terms(subset)
    if not terms:
        return None
    terms = list(reversed(terms))
    term_idx = {term: i for i, term in enumerate(terms)}
    cat_idx = CATEGORY_X

    xs = []
    ys = []
    sizes = []
    colors = []
    for row in subset:
        term = str(row["term_with_library"])
        category = str(row["category"])
        if term not in term_idx or category not in cat_idx:
            continue
        xs.append(cat_idx[category])
        ys.append(term_idx[term])
        score = plot_score(row)
        overlap_n = int(row.get("overlap_count", 0))
        sizes.append(max(45.0, min(520.0, overlap_n * 24.0)))
        colors.append(score)

    if not xs:
        return None

    width = 10.5
    height = max(5.5, 0.55 * len(terms) + 2.0)
    fig, ax = plt.subplots(figsize=(width, height))
    max_color = max(max(colors), 1.0)
    sc = ax.scatter(
        xs,
        ys,
        s=sizes,
        c=colors,
        cmap="Reds",
        vmin=0,
        vmax=max_color,
        marker="o",
        edgecolors="0.35",
        linewidths=0.7,
        alpha=0.9,
    )
    ax.set_xticks([CATEGORY_X[c] for c in CATEGORY_ORDER])
    ax.set_xticklabels([CATEGORY_LABELS[c] for c in CATEGORY_ORDER], fontsize=11, fontweight="bold")
    ax.set_yticks(range(len(terms)))
    ax.set_yticklabels([wrap_label(t) for t in terms], fontsize=8)
    ax.set_xlim(-0.8, 0.8)
    ax.set_xlabel("Analysis group", fontweight="bold")
    score_label = "-log10(FDR)" if PLOT_SCORE == "fdr" else "-log10(raw p-value)"
    ax.set_title(f"DEG enrichment: {metric}\n{direction.replace('_', ' ')}", fontsize=12)
    ax.grid(axis="x", alpha=0.18)
    ax.grid(axis="y", alpha=0.10)

    cbar = plt.colorbar(sc, ax=ax, pad=0.02)
    cbar.set_label(score_label, fontweight="bold")

    size_handles = [
        ax.scatter([], [], s=max(45.0, min(520.0, n * 24.0)), color="white", edgecolors="0.35", label=f"{n} genes")
        for n in (2, 5, 10)
    ]
    ax.legend(handles=size_handles, title="DEG overlap", loc="lower right", frameon=False)

    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"combined_enrichr_{safe_text(metric)}_{safe_text(direction)}.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return str(out_path)


def main() -> None:
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    rows = load_all_results()
    combined_csv = EXPORT_DIR / "combined_enrichr_results_global_breast_prostate.csv"
    write_combined_csv(rows, combined_csv)

    plot_rows = []
    metrics = sorted({str(r["metric"]) for r in rows})
    directions = [d for d in DIRECTION_ORDER if any(str(r["direction"]) == d for r in rows)]
    for metric in metrics:
        for direction in directions:
            plot_path = plot_metric_direction(rows, metric, direction, EXPORT_DIR / "plots")
            if plot_path:
                plot_rows.append({"metric": metric, "direction": direction, "plot": plot_path})
                print(f"plotted: {metric} / {direction}")

    plot_index = EXPORT_DIR / "combined_enrichr_plot_index.csv"
    with plot_index.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["metric", "direction", "plot"])
        writer.writeheader()
        writer.writerows(plot_rows)

    print("Wrote:")
    print(combined_csv)
    print(plot_index)
    print(EXPORT_DIR / "plots")


if __name__ == "__main__":
    main()
