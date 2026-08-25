#!/usr/bin/env python3
"""Reactome enrichment of model-predicted changes; observed post is never used."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import gseapy as gp
import numpy as np
import pandas as pd

from RNA_validation.common import clean, require_empty_output, write_json


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--prediction-dir", type=Path, required=True)
    p.add_argument("--reactome-gmt", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--permutations", type=int, default=1000)
    p.add_argument("--min-size", type=int, default=15)
    p.add_argument("--max-size", type=int, default=500)
    p.add_argument("--minimum-pathway-coverage", type=float, default=0.20)
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_gmt(path: Path) -> dict[str, list[str]]:
    pathways: dict[str, list[str]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 3:
                continue
            genes = list(dict.fromkeys(clean(x).upper() for x in fields[2:] if clean(x)))
            if genes:
                pathways[clean(fields[0])] = genes
    if not pathways:
        raise ValueError(f"No gene sets found in {path}")
    return pathways


def load_gene_table(path: Path, metadata: dict[str, object]) -> pd.DataFrame:
    frame = pd.read_csv(path, sep="\t")
    measured = frame["measured_pre"].astype(str).str.lower().eq("true")
    frame = frame.loc[measured, ["gene_symbol", "predicted_delta"]].copy()
    frame["gene_symbol"] = frame["gene_symbol"].map(clean).str.upper()
    frame["predicted_delta"] = pd.to_numeric(frame["predicted_delta"], errors="coerce")
    frame = frame.loc[frame["gene_symbol"].ne("") & frame["predicted_delta"].notna()]
    for key, value in metadata.items():
        frame[key] = value
    return frame


def contexts(prediction_dir: Path) -> list[tuple[str, str, pd.DataFrame]]:
    manifest = pd.read_csv(prediction_dir / "prediction_manifest.tsv", sep="\t", dtype=str, keep_default_na=False)
    parts = []
    for row in manifest.itertuples(index=False):
        parts.append(load_gene_table(prediction_dir / row.gene_table, {
            "sample_id": row.sample_id, "patient_key": row.patient_key,
            "gse_id": row.gse_id, "response_group": row.response_group,
        }))
    long = pd.concat(parts, ignore_index=True)
    result: list[tuple[str, str, pd.DataFrame]] = [("global", "all_baselines", long)]
    for kind, column in (("dataset", "gse_id"), ("response", "response_group")):
        for label, group in long.groupby(column, dropna=False):
            if group["patient_key"].nunique() >= 3 and clean(label).lower() not in {"", "na", "mixed"}:
                result.append((kind, clean(label), group))

    segment_path = prediction_dir / "segment_prediction_manifest.tsv"
    if segment_path.exists():
        segments = pd.read_csv(segment_path, sep="\t", dtype=str, keep_default_na=False)
        segment_parts = []
        for row in segments.itertuples(index=False):
            segment_parts.append(load_gene_table(prediction_dir / row.segment_gene_table, {
                "sample_id": row.sample_id, "patient_key": row.patient_key,
                "gse_id": row.gse_id, "response_group": row.response_group,
                "segment_type": row.segment_type,
            }))
        if segment_parts:
            segment_long = pd.concat(segment_parts, ignore_index=True)
            for label, group in segment_long.groupby("segment_type", dropna=False):
                if group["patient_key"].nunique() >= 3:
                    result.append(("segment", clean(label) or "all", group))
    return result


def make_rank(frame: pd.DataFrame) -> pd.DataFrame:
    rank = frame.groupby("gene_symbol", as_index=False)["predicted_delta"].mean()
    rank = rank.replace([np.inf, -np.inf], np.nan).dropna()
    rank = rank.sort_values("predicted_delta", ascending=False)
    return rank


def main() -> None:
    args = parse_args()
    out = require_empty_output(args.output_dir, args.overwrite)
    pathways = read_gmt(args.reactome_gmt)
    all_results, context_rows, errors = [], [], []

    for context_type, context, frame in contexts(args.prediction_dir):
        rank = make_rank(frame)
        ranked_genes = set(rank["gene_symbol"])
        retained: dict[str, list[str]] = {}
        coverage_values = []
        for term, genes in pathways.items():
            overlap = ranked_genes.intersection(genes)
            coverage = len(overlap) / len(genes)
            if coverage >= args.minimum_pathway_coverage and args.min_size <= len(overlap) <= args.max_size:
                retained[term] = genes
                coverage_values.append(coverage)
        context_rows.append({
            "context_type": context_type, "context": context,
            "samples": int(frame["sample_id"].nunique()),
            "patients": int(frame["patient_key"].nunique()),
            "ranked_genes": len(rank), "reactome_pathways_retained": len(retained),
            "median_pathway_gene_coverage": float(np.median(coverage_values)) if coverage_values else np.nan,
        })
        if not retained:
            errors.append({"context_type": context_type, "context": context, "error": "no pathways passed coverage/size filters"})
            continue
        try:
            result = gp.prerank(
                rnk=rank[["gene_symbol", "predicted_delta"]], gene_sets=retained,
                permutation_num=args.permutations, min_size=args.min_size,
                max_size=args.max_size, seed=args.seed, threads=args.threads,
                outdir=None, verbose=False,
            ).res2d.reset_index(drop=True)
            result.insert(0, "context", context)
            result.insert(0, "context_type", context_type)
            all_results.append(result)
        except Exception as exc:
            errors.append({"context_type": context_type, "context": context, "error": str(exc)})

    results = pd.concat(all_results, ignore_index=True) if all_results else pd.DataFrame()
    results.to_csv(out / "reactome_prerank_results.tsv", sep="\t", index=False)
    pd.DataFrame(context_rows).to_csv(out / "reactome_contexts.tsv", sep="\t", index=False)
    pd.DataFrame(errors, columns=["context_type", "context", "error"]).to_csv(
        out / "reactome_errors.tsv", sep="\t", index=False
    )
    write_json(out / "reactome_run_metadata.json", {
        "reactome_gmt": str(args.reactome_gmt),
        "reactome_gmt_sha256": sha256(args.reactome_gmt),
        "reactome_gene_sets": len(pathways),
        "permutations": args.permutations,
        "minimum_pathway_coverage": args.minimum_pathway_coverage,
        "contrast": "pretreatment versus model-predicted posttreatment",
        "measured_posttreatment_used": False,
        "contexts_tested": len(context_rows),
        "contexts_failed": len(errors),
    })
    print(f"reactome_contexts={len(context_rows)} result_rows={len(results)} output={out}")


if __name__ == "__main__":
    main()
