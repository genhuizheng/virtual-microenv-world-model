#!/usr/bin/env python3
import csv
import math
import shutil
from pathlib import Path

# ======================
# Local variables
# ======================
RESULT_DIR = Path(r"C:\Users\ncsub\Downloads\initial_biological_analysis_global")
EXPORT_DIR = RESULT_DIR / "screen_breast_prostate_global_export"

# Use None for all metrics, or set e.g. "combined_concentration"
METRIC = None

BREAST_KEYS = ("breast", "mammary")
PROSTATE_KEYS = ("prostate", "prostatic", "prostrate")

def read_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))

def fnum(x):
    if x in ("", None, "nan", "NaN", "None"):
        return math.nan
    return float(x)

def match_category(scope, tumor_type):
    text = str(tumor_type).lower()
    if str(scope).lower() == "global" or text == "all":
        return "global"
    if any(k in text for k in BREAST_KEYS):
        return "breast"
    if any(k in text for k in PROSTATE_KEYS):
        return "prostate"
    return None

def gene_name(row):
    for k in ("gene_symbol", "gene", "gene_id", "index"):
        if row.get(k):
            return row[k]
    return ""

def sort_gene_rows(rows):
    def key(r):
        fdr = fnum(r.get("paired_t_fdr_bh", "nan"))
        p = fnum(r.get("paired_t_pvalue", "nan"))
        delta = abs(fnum(r.get("mean_delta", "nan")))
        return (
            math.inf if math.isnan(fdr) else fdr,
            math.inf if math.isnan(p) else p,
            -(0 if math.isnan(delta) else delta),
        )
    return sorted(rows, key=key)

def safe_text(x):
    return "".join(c if c.isalnum() or c in "._-" else "_" for c in str(x)).strip("_")[:100] or "unknown"

def localize_path(path_text, result_dir):
    p = Path(path_text)
    if p.exists():
        return p
    p2 = result_dir / p.name
    if p2.exists():
        return p2
    return None

def copy_plots_for_result(result_dir, export_dir, category, scope, tumor_type, metric):
    plots_dir = result_dir / "plots"
    if not plots_dir.exists():
        return []

    safe_tumor = safe_text(tumor_type)
    safe_metric = safe_text(metric)

    dest_dir = export_dir / "plots" / category / safe_metric
    dest_dir.mkdir(parents=True, exist_ok=True)

    copied = []
    for p in plots_dir.glob("*.png"):
        name = p.name
        if safe_metric not in name:
            continue
        if category == "global" and "global_all" not in name:
            continue
        if category != "global" and safe_tumor not in name:
            continue

        dest = dest_dir / p.name
        shutil.copy2(p, dest)
        copied.append(str(dest))

    return copied

def main():
    result_dir = RESULT_DIR
    export_dir = EXPORT_DIR
    export_dir.mkdir(parents=True, exist_ok=True)

    summary_path = result_dir / "formal_de_outputs_by_tumor_metric.csv"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing: {summary_path}")

    selected = []
    top_gene_rows = []
    plot_rows = []

    for row in read_csv(summary_path):
        scope = row.get("scope", "")
        tumor_type = row.get("tumor_type", "")
        metric = row.get("metric", "")

        if METRIC is not None and metric != METRIC:
            continue

        category = match_category(scope, tumor_type)
        if category is None:
            continue

        selected.append({
            "category": category,
            "scope": scope,
            "tumor_type": tumor_type,
            "metric": metric,
            "n_pairs": row.get("n_pairs", ""),
            "n_replicates": row.get("n_replicates", ""),
            "n_fdr_0p05": row.get("n_fdr_0p05", ""),
            "n_fdr_0p10": row.get("n_fdr_0p10", ""),
            "top_gene_csv": row.get("top_gene_csv", ""),
            "gene_csv": row.get("gene_csv", ""),
        })

        top_path = localize_path(row.get("top_gene_csv", ""), result_dir)
        if top_path:
            genes = sort_gene_rows(read_csv(top_path))  # extract all genes
            for rank, g in enumerate(genes, 1):
                top_gene_rows.append({
                    "category": category,
                    "tumor_type": tumor_type,
                    "metric": metric,
                    "gene_rank": rank,
                    "gene": gene_name(g),
                    "mean_delta": g.get("mean_delta", ""),
                    "fdr": g.get("paired_t_fdr_bh", ""),
                    "pvalue": g.get("paired_t_pvalue", ""),
                    "statistic": g.get("paired_t_statistic", ""),
                })

        for plot_path in copy_plots_for_result(result_dir, export_dir, category, scope, tumor_type, metric):
            plot_rows.append({
                "category": category,
                "tumor_type": tumor_type,
                "metric": metric,
                "plot": plot_path,
            })

    selected.sort(key=lambda r: (
        {"global": 0, "breast": 1, "prostate": 2}.get(r["category"], 9),
        r["metric"],
        r["tumor_type"],
    ))

    outputs = [
        (export_dir / "breast_prostate_global_summary.csv", selected),
        (export_dir / "breast_prostate_global_all_genes.csv", top_gene_rows),
        (export_dir / "breast_prostate_global_plots.csv", plot_rows),
    ]

    for path, rows in outputs:
        if not rows:
            path.write_text("", encoding="utf-8")
            continue
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    print("Wrote:")
    for path, _ in outputs:
        print(path)
    print(export_dir / "plots")

if __name__ == "__main__":
    main()