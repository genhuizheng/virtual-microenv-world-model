#!/usr/bin/env python3
"""Run DEG-based GO/KEGG/Reactome enrichment for global, breast, and prostate."""

from __future__ import annotations

import csv
import math
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


# ======================
# Local variables
# ======================
RESULT_DIR = Path(r"C:\Users\ncsub\Downloads\initial_biological_analysis_global")
EXPORT_DIR = RESULT_DIR / "screen_breast_prostate_global_deg_enrichment"

# Use None for all confidence metrics, or set e.g. "combined_concentration"
METRIC = None

FDR_CUTOFF = 0.05
MIN_GENES = 5
PLOT_FDR_CUTOFF = 0.25
TOP_TERMS_PER_PLOT = 12
MAX_WORKERS = 1
SKIP_EXISTING = True
ENRICHR_RETRIES = 5
ENRICHR_RETRY_SLEEP_SECONDS = 30

ENRICHR_LIBRARIES = [
    "GO_Biological_Process_2023",
    "KEGG_2021_Human",
    "Reactome_2022",
]

BREAST_KEYS = ("breast", "mammary")
PROSTATE_KEYS = ("prostate", "prostatic", "prostrate")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def safe_text(x: object) -> str:
    return "".join(c if c.isalnum() or c in "._-" else "_" for c in str(x)).strip("_")[:100] or "unknown"


def pretty_library_name(library: str) -> str:
    if library.startswith("KEGG"):
        return "KEGG"
    if library.startswith("Reactome"):
        return "Reactome"
    if library.startswith("GO_"):
        return "GO BP"
    return library.split("_")[0]


def fnum(x: object) -> float:
    if x in ("", None, "nan", "NaN", "None"):
        return math.nan
    try:
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


def gene_symbol(row: dict[str, str]) -> str:
    for key in ("gene_symbol", "gene", "gene_name", "gene_id", "index"):
        value = str(row.get(key, "")).strip()
        if not value:
            continue
        upper = value.upper()
        if upper.startswith("ENSG"):
            continue
        return upper
    return ""


def delta_value(row: dict[str, str]) -> float:
    for key in ("pseudobulk_paired_delta", "mean_delta_target_minus_source", "mean_delta"):
        value = fnum(row.get(key))
        if math.isfinite(value):
            return value
    return math.nan


def split_deg_genes(gene_csv: Path) -> dict[str, list[str]]:
    all_degs: list[str] = []
    up: list[str] = []
    down: list[str] = []

    seen = {"all": set(), "up": set(), "down": set()}
    for row in read_csv(gene_csv):
        fdr = fnum(row.get("paired_t_fdr_bh"))
        if not math.isfinite(fdr) or fdr > FDR_CUTOFF:
            continue

        gene = gene_symbol(row)
        if not gene:
            continue

        delta = delta_value(row)

        if gene not in seen["all"]:
            all_degs.append(gene)
            seen["all"].add(gene)

        if math.isfinite(delta) and delta > 0 and gene not in seen["up"]:
            up.append(gene)
            seen["up"].add(gene)
        elif math.isfinite(delta) and delta < 0 and gene not in seen["down"]:
            down.append(gene)
            seen["down"].add(gene)

    return {
        "all_degs": all_degs,
        "up_after_treatment": up,
        "down_after_treatment": down,
    }


def run_enrichr(genes: list[str], library: str, out_dir: Path) -> Path | None:
    if len(genes) < MIN_GENES:
        return None
    result_path = out_dir / "enrichr_results.csv"
    if SKIP_EXISTING and result_path.exists() and result_path.stat().st_size > 0:
        return result_path

    try:
        import gseapy as gp
    except ImportError as exc:
        raise SystemExit("Missing package: gseapy. Install with: pip install gseapy") from exc

    out_dir.mkdir(parents=True, exist_ok=True)
    enr = None
    for attempt in range(1, ENRICHR_RETRIES + 1):
        try:
            enr = gp.enrichr(
                gene_list=genes,
                gene_sets=library,
                organism="human",
                outdir=str(out_dir),
                cutoff=1.0,
                no_plot=False,
                verbose=False,
            )
            break
        except Exception as exc:
            if attempt >= ENRICHR_RETRIES:
                raise
            sleep_seconds = ENRICHR_RETRY_SLEEP_SECONDS * attempt
            print(f"Enrichr retry {attempt}/{ENRICHR_RETRIES} for {library}: {exc}. Sleeping {sleep_seconds}s")
            time.sleep(sleep_seconds)
    result = getattr(enr, "results", None)
    if result is not None:
        result.to_csv(result_path, index=False)
        return result_path
    candidates = sorted(out_dir.glob("*.csv"))
    return candidates[0] if candidates else None


def run_enrichment_job(job: dict[str, object]) -> dict[str, str]:
    genes = job["genes"]
    library = str(job["library"])
    out_dir = Path(str(job["out_dir"]))
    result_path = run_enrichr(genes, library, out_dir)  # type: ignore[arg-type]
    return {
        "category": str(job["category"]),
        "tumor_type": str(job["tumor_type"]),
        "metric": str(job["metric"]),
        "direction": str(job["direction"]),
        "library": library,
        "n_genes": str(job["n_genes"]),
        "gene_list": str(job["gene_list"]),
        "result_path": str(result_path or ""),
        "out_dir": str(out_dir),
    }


def adjusted_p_value(row: dict[str, str]) -> float:
    for key in ("Adjusted P-value", "Adjusted P Value", "Adjusted_P_value", "FDR", "fdr"):
        value = fnum(row.get(key))
        if math.isfinite(value):
            return value
    return math.nan


def clean_term(term: str) -> str:
    term = str(term)
    # Remove GO identifiers when present to keep y-axis readable.
    if "(GO:" in term:
        term = term.split("(GO:", 1)[0].strip()
    return term[:95]


def read_enrichment_terms(index_rows: list[dict[str, str]]) -> list[dict[str, object]]:
    terms = []
    for idx in index_rows:
        direction = idx["direction"]
        if direction not in ("up_after_treatment", "down_after_treatment"):
            continue
        result_path = Path(idx.get("result_path", ""))
        if not result_path.exists():
            continue

        sign = 1.0 if direction == "up_after_treatment" else -1.0
        for row in read_csv(result_path):
            fdr = adjusted_p_value(row)
            if not math.isfinite(fdr) or fdr <= 0 or fdr > PLOT_FDR_CUTOFF:
                continue
            score = sign * (-math.log10(max(fdr, 1e-300)))
            term = row.get("Term", "")
            if not term:
                continue
            terms.append(
                {
                    "category": idx["category"],
                    "tumor_type": idx["tumor_type"],
                    "metric": idx["metric"],
                    "library": idx["library"],
                    "direction": direction,
                    "term": clean_term(term),
                    "plot_term": f"{clean_term(term)} [{pretty_library_name(idx['library'])}]",
                    "fdr": fdr,
                    "signed_log10_fdr": score,
                    "size_value": -math.log10(max(fdr, 1e-300)),
                }
            )
    return terms


def plot_directional_enrichment(terms: list[dict[str, object]], export_dir: Path) -> list[dict[str, str]]:
    if not terms:
        return []

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit("Missing package: matplotlib. Install with: pip install matplotlib") from exc

    plot_rows = []
    grouped: dict[tuple[str, str, str], list[dict[str, object]]] = {}
    for term in terms:
        key = (str(term["category"]), str(term["tumor_type"]), str(term["metric"]))
        grouped.setdefault(key, []).append(term)

    plot_dir = export_dir / "directional_enrichment_plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    for (category, tumor_type, metric), rows in grouped.items():
        # Keep strongest terms by absolute signed enrichment score.
        rows = sorted(rows, key=lambda r: abs(float(r["signed_log10_fdr"])), reverse=True)[:TOP_TERMS_PER_PLOT]
        rows = list(reversed(rows))

        y_labels = [str(r["plot_term"]) for r in rows]
        x = [float(r["signed_log10_fdr"]) for r in rows]
        sizes = [max(30.0, float(r["size_value"]) * 70.0) for r in rows]
        y = list(range(len(rows)))
        max_abs = max(max(abs(v) for v in x), 1.0)

        width = 9.0
        height = max(4.5, 0.45 * len(rows) + 1.8)
        fig, ax = plt.subplots(figsize=(width, height))
        scatter = ax.scatter(
            x,
            y,
            s=sizes,
            c=x,
            cmap="coolwarm",
            vmin=-max_abs,
            vmax=max_abs,
            edgecolors="0.45",
            linewidths=0.6,
            alpha=0.9,
        )
        ax.axvline(0, color="black", linestyle="--", linewidth=1.2)
        ax.set_yticks(y)
        ax.set_yticklabels(y_labels, fontsize=9)
        ax.set_xlabel("Signed -log10(FDR) enrichment score\n(up after treatment + / down after treatment -)", fontsize=12, fontweight="bold")
        ax.set_title(f"DEG enrichment: {tumor_type} / {metric}", fontsize=12)
        ax.grid(axis="x", alpha=0.2)
        for spine in ax.spines.values():
            spine.set_linewidth(1.1)

        cbar = plt.colorbar(scatter, ax=ax, pad=0.03)
        cbar.set_label("Signed -log10(FDR)", fontweight="bold")

        # Size legend similar to the example figure.
        legend_fdrs = [0.25, 0.05, 0.001]
        handles = []
        labels = []
        for fdr in legend_fdrs:
            value = -math.log10(fdr)
            handles.append(ax.scatter([], [], s=max(30.0, value * 70.0), color="lightgray", edgecolors="0.55"))
            labels.append(f"FDR {fdr:g}")
        ax.legend(handles, labels, title="-log10(FDR) size", loc="lower right", frameon=False)

        fig.tight_layout()
        out_path = plot_dir / f"directional_deg_enrichment_{safe_text(category)}_{safe_text(tumor_type)}_{safe_text(metric)}.png"
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

        plot_rows.append(
            {
                "category": category,
                "tumor_type": tumor_type,
                "metric": metric,
                "plot": str(out_path),
            }
        )

    return plot_rows


def main() -> None:
    result_dir = RESULT_DIR
    export_dir = EXPORT_DIR
    export_dir.mkdir(parents=True, exist_ok=True)

    summary_path = result_dir / "formal_de_outputs_by_tumor_metric.csv"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing: {summary_path}")

    index_rows = []
    deg_count_rows = []
    jobs = []

    for row in read_csv(summary_path):
        scope = row.get("scope", "")
        tumor_type = row.get("tumor_type", "")
        metric = row.get("metric", "")
        if METRIC is not None and metric != METRIC:
            continue

        category = match_category(scope, tumor_type)
        if category is None:
            continue

        gene_csv = localize_path(row.get("gene_csv", ""), result_dir)
        if gene_csv is None:
            continue

        deg_sets = split_deg_genes(gene_csv)
        label = f"{category}_{safe_text(tumor_type)}_{safe_text(metric)}"

        for direction, genes in deg_sets.items():
            gene_list_path = export_dir / "deg_gene_lists" / f"{label}_{direction}.txt"
            gene_list_path.parent.mkdir(parents=True, exist_ok=True)
            gene_list_path.write_text("\n".join(genes) + ("\n" if genes else ""), encoding="utf-8")

            deg_count_rows.append(
                {
                    "category": category,
                    "tumor_type": tumor_type,
                    "metric": metric,
                    "direction": direction,
                    "n_genes": len(genes),
                    "gene_list": str(gene_list_path),
                }
            )

            if len(genes) < MIN_GENES:
                continue

            for library in ENRICHR_LIBRARIES:
                out_dir = export_dir / "enrichment_results" / label / direction / safe_text(library)
                jobs.append(
                    {
                        "category": category,
                        "tumor_type": tumor_type,
                        "metric": metric,
                        "direction": direction,
                        "library": library,
                        "n_genes": len(genes),
                        "gene_list": str(gene_list_path),
                        "genes": genes,
                        "out_dir": out_dir,
                    }
                )

    print(f"Prepared enrichment jobs: {len(jobs)}")
    print(f"Parallel workers: {MAX_WORKERS}")
    if jobs:
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            future_map = {executor.submit(run_enrichment_job, job): job for job in jobs}
            for i, future in enumerate(as_completed(future_map), 1):
                job = future_map[future]
                label = f"{job['category']}_{safe_text(job['tumor_type'])}_{safe_text(job['metric'])}"
                try:
                    row = future.result()
                    index_rows.append(row)
                    print(f"[{i}/{len(jobs)}] done {label} / {job['direction']} / {job['library']}")
                except Exception as exc:
                    print(f"[{i}/{len(jobs)}] failed {label} / {job['direction']} / {job['library']}: {exc}")

    deg_counts_path = export_dir / "deg_gene_counts.csv"
    with deg_counts_path.open("w", newline="", encoding="utf-8") as f:
        fields = ["category", "tumor_type", "metric", "direction", "n_genes", "gene_list"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(deg_count_rows)

    index_path = export_dir / "deg_enrichment_run_index.csv"
    with index_path.open("w", newline="", encoding="utf-8") as f:
        fields = ["category", "tumor_type", "metric", "direction", "library", "n_genes", "gene_list", "result_path", "out_dir"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(index_rows)

    terms = read_enrichment_terms(index_rows)
    terms_path = export_dir / "deg_enrichment_directional_terms_for_plot.csv"
    with terms_path.open("w", newline="", encoding="utf-8") as f:
        fields = ["category", "tumor_type", "metric", "library", "direction", "term", "fdr", "signed_log10_fdr"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in terms:
            writer.writerow({field: row[field] for field in fields})

    plot_rows = plot_directional_enrichment(terms, export_dir)
    plots_path = export_dir / "deg_enrichment_directional_plots.csv"
    with plots_path.open("w", newline="", encoding="utf-8") as f:
        fields = ["category", "tumor_type", "metric", "plot"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(plot_rows)

    print("Wrote:")
    print(deg_counts_path)
    print(index_path)
    print(terms_path)
    print(plots_path)
    print(export_dir / "deg_gene_lists")
    print(export_dir / "enrichment_results")
    print(export_dir / "directional_enrichment_plots")


if __name__ == "__main__":
    main()
