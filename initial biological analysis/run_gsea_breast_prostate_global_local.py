#!/usr/bin/env python3
"""Run local GSEA for global, breast, and prostate DE outputs."""

from __future__ import annotations

import csv
import math
from pathlib import Path


# ======================
# Local variables
# ======================
RESULT_DIR = Path(r"C:\Users\ncsub\Downloads\initial_biological_analysis_global")
EXPORT_DIR = RESULT_DIR / "screen_breast_prostate_global_gsea"

# Use None for all confidence metrics, or set e.g. "combined_concentration"
METRIC = None

# Public Enrichr/GSEApy gene-set libraries.
# Requires internet access the first time GSEApy downloads them.
GENE_SET_LIBRARIES = [
    "GO_Biological_Process_2023",
    "KEGG_2021_Human",
    "Reactome_2022",
    "MSigDB_Hallmark_2020",
]

PERMUTATION_NUM = 1000
MIN_SIZE = 10
MAX_SIZE = 500
SEED = 0

BREAST_KEYS = ("breast", "mammary")
PROSTATE_KEYS = ("prostate", "prostatic", "prostrate")


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


def gene_name(row: dict[str, str]) -> str:
    # Public Enrichr/GSEApy libraries use HGNC-style gene symbols, not Ensembl IDs.
    for key in ("gene_symbol", "gene", "gene_name", "gene_id", "index"):
        value = str(row.get(key, "")).strip()
        if not value:
            continue
        upper = value.upper()
        if upper.startswith("ENSG"):
            continue
        return upper
    return ""


def ranking_score(row: dict[str, str]) -> float:
    stat = fnum(row.get("paired_t_stat"))
    if math.isfinite(stat):
        return stat
    delta = fnum(row.get("pseudobulk_paired_delta"))
    if math.isfinite(delta):
        return delta
    delta = fnum(row.get("mean_delta_target_minus_source"))
    if math.isfinite(delta):
        return delta
    logfc = fnum(row.get("pseudobulk_log2fc"))
    if math.isfinite(logfc):
        return logfc
    return math.nan


def make_rank_file(gene_csv: Path, out_path: Path) -> int:
    rows = read_csv(gene_csv)
    best_by_gene: dict[str, float] = {}
    for row in rows:
        gene = gene_name(row)
        score = ranking_score(row)
        if not gene or not math.isfinite(score):
            continue
        if gene not in best_by_gene or abs(score) > abs(best_by_gene[gene]):
            best_by_gene[gene] = score

    ranked = sorted(best_by_gene.items(), key=lambda x: x[1], reverse=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        # GSEApy prerank expects a two-column rank file without a header.
        writer.writerows(ranked)
    return len(ranked)


def run_one_gsea(rank_path: Path, gene_sets: str, out_dir: Path):
    try:
        import gseapy as gp
    except ImportError as exc:
        raise SystemExit("Missing package: gseapy. Install with: pip install gseapy") from exc

    out_dir.mkdir(parents=True, exist_ok=True)
    return gp.prerank(
        rnk=str(rank_path),
        gene_sets=gene_sets,
        outdir=str(out_dir),
        permutation_num=PERMUTATION_NUM,
        min_size=MIN_SIZE,
        max_size=MAX_SIZE,
        seed=SEED,
        no_plot=False,
        verbose=False,
    )


def find_result_table(gsea_dir: Path) -> Path | None:
    candidates = list(gsea_dir.glob("*.csv")) + list(gsea_dir.glob("*.tsv")) + list(gsea_dir.glob("*.res2d.txt"))
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_size, reverse=True)
    return candidates[0]


def main() -> None:
    result_dir = RESULT_DIR
    export_dir = EXPORT_DIR
    export_dir.mkdir(parents=True, exist_ok=True)

    summary_path = result_dir / "formal_de_outputs_by_tumor_metric.csv"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing: {summary_path}")

    runs = []
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

        safe_label = f"{category}_{safe_text(tumor_type)}_{safe_text(metric)}"
        rank_path = export_dir / "rank_files" / f"{safe_label}.rnk"
        n_genes = make_rank_file(gene_csv, rank_path)
        if n_genes < MIN_SIZE:
            continue

        for library in GENE_SET_LIBRARIES:
            gsea_dir = export_dir / "gsea_results" / safe_label / safe_text(library)
            print(f"Running GSEA: {safe_label} / {library} / n_genes={n_genes}")
            run_one_gsea(rank_path, library, gsea_dir)
            runs.append(
                {
                    "category": category,
                    "tumor_type": tumor_type,
                    "metric": metric,
                    "library": library,
                    "n_ranked_genes": n_genes,
                    "rank_file": str(rank_path),
                    "gsea_dir": str(gsea_dir),
                    "result_table": str(find_result_table(gsea_dir) or ""),
                }
            )

    index_path = export_dir / "gsea_run_index.csv"
    with index_path.open("w", newline="", encoding="utf-8") as f:
        fields = ["category", "tumor_type", "metric", "library", "n_ranked_genes", "rank_file", "gsea_dir", "result_table"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(runs)

    print("Wrote:")
    print(index_path)
    print(export_dir / "gsea_results")
    print(export_dir / "rank_files")


if __name__ == "__main__":
    main()
