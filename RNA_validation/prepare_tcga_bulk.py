#!/usr/bin/env python3
"""Select one TCGA tumor profile per patient and align it to checkpoint gene order."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from RNA_validation.common import (
    GeneAligner,
    checkpoint_gene_table,
    load_checkpoint,
    read_tsv,
    require_empty_output,
    transform_counts,
    write_json,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input-dir", type=Path, required=True, help="standardized_tcga_raw_counts")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--model-genes", type=Path, default=None)
    p.add_argument("--ranked-reference", type=Path, default=None)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--transform", choices=["auto", "none", "cpm", "log1p_cpm", "log1p_10k"], default="auto")
    p.add_argument("--include-ffpe", action="store_true")
    p.add_argument("--include-nonprimary", action="store_true")
    p.add_argument("--limit-projects", type=int, default=None, help="Deterministic smoke-test limit after sorting projects")
    p.add_argument("--limit-patients-per-project", type=int, default=None, help="Deterministic smoke-test patient limit")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def select_files(input_dir: Path, include_ffpe: bool, include_nonprimary: bool) -> pd.DataFrame:
    samples = read_tsv(input_dir / "samples.tsv")
    files = read_tsv(input_dir / "expression_files.tsv")
    table = files.merge(samples, on=["sample_id", "patient_id", "dataset_id", "source_project_id"], how="left", validate="many_to_one")
    table = table.loc[table["tissue_type_source"].eq("Tumor")].copy()
    if not include_nonprimary:
        table = table.loc[table["tumor_descriptor_source"].eq("Primary")].copy()
    if not include_ffpe:
        table = table.loc[~table["preservation_method_source"].eq("FFPE")].copy()
    table = table.sort_values(
        ["dataset_id", "patient_id", "sample_id", "source_file_id"], kind="stable"
    ).drop_duplicates("patient_id", keep="first")
    if table.empty:
        raise ValueError("TCGA sample filters selected no files")
    return table


def main() -> None:
    args = parse_args()
    for name in ("limit_projects", "limit_patients_per_project"):
        value = getattr(args, name)
        if value is not None and value < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    out = require_empty_output(args.output_dir, args.overwrite)
    shards = out / "shards"
    shards.mkdir(exist_ok=True)
    checkpoint = load_checkpoint(args.checkpoint)
    transform = (
        str(checkpoint.get("args", {}).get("expression_transform", "none"))
        if args.transform == "auto" else args.transform
    )
    model_genes = checkpoint_gene_table(checkpoint, args.model_genes)
    aligner = GeneAligner(model_genes, args.ranked_reference)
    selected = select_files(args.input_dir, args.include_ffpe, args.include_nonprimary)
    if args.limit_projects is not None:
        projects = sorted(selected["dataset_id"].unique())[:args.limit_projects]
        selected = selected.loc[selected["dataset_id"].isin(projects)].copy()
    if args.limit_patients_per_project is not None:
        selected = selected.groupby("dataset_id", sort=True, group_keys=False).head(args.limit_patients_per_project).copy()
    selected.to_csv(out / "selected_expression_files.tsv", sep="\t", index=False)
    model_genes.to_csv(out / "model_genes.tsv", sep="\t", index=False)

    manifests = []
    audit_rows = []
    for dataset_id, chosen in selected.groupby("dataset_id", sort=True):
        matrix_path = args.input_dir / "counts" / f"{dataset_id}_unstranded_raw_counts.tsv"
        requested = chosen["raw_count_matrix_column_id"].tolist()
        wanted = set(requested)
        usecols = lambda c: c in {"gene_id", "gene_id_versioned", "gene_symbol_source", "gene_type_source"} or c in wanted
        matrix = pd.read_csv(matrix_path, sep="\t", usecols=usecols, dtype={"gene_id": str, "gene_symbol_source": str})
        missing = wanted - set(matrix.columns)
        if missing:
            raise ValueError(f"Missing {len(missing)} selected columns in {matrix_path}")
        raw = matrix[requested].to_numpy(dtype=np.float32)
        if np.any(raw < 0) or not np.all(np.isfinite(raw)):
            raise ValueError(f"Invalid counts in {matrix_path}")
        library_sizes = raw.sum(axis=0, dtype=np.float64)
        alignment = aligner.resolve(matrix["gene_id"], matrix["gene_symbol_source"])
        aligned = aligner.align_gene_by_sample(raw, alignment)
        x = transform_counts(aligned, library_sizes, transform)
        shard_name = f"{dataset_id}.npz"
        np.savez_compressed(shards / shard_name, x_pre=x)
        chosen = chosen.set_index("raw_count_matrix_column_id").loc[requested].reset_index()
        chosen.insert(0, "row_index", np.arange(len(chosen), dtype=int))
        meta_name = f"{dataset_id}_patients.tsv"
        chosen.to_csv(shards / meta_name, sep="\t", index=False)
        observed = set(alignment.source_to_model[alignment.source_to_model >= 0])
        methods = pd.Series(alignment.method).value_counts().to_dict()
        audit_rows.append({
            "dataset_id": dataset_id,
            "source_genes": len(alignment.source_to_model),
            "mapped_source_rows": int((alignment.source_to_model >= 0).sum()),
            "unique_model_genes_observed": len(observed),
            "model_genes": len(model_genes),
            "model_gene_coverage": len(observed) / len(model_genes),
            **{f"mapping_{k}": int(v) for k, v in methods.items()},
        })
        manifests.append({
            "dataset_id": dataset_id,
            "expression_npz": f"shards/{shard_name}",
            "metadata_tsv": f"shards/{meta_name}",
            "n_patients": len(chosen),
            "n_model_genes": len(model_genes),
        })
        print(f"prepared {dataset_id}: patients={len(chosen)} coverage={len(observed)/len(model_genes):.3f}")

    pd.DataFrame(manifests).to_csv(out / "manifest.tsv", sep="\t", index=False)
    pd.DataFrame(audit_rows).to_csv(out / "gene_alignment_summary.tsv", sep="\t", index=False)
    write_json(out / "manifest.json", {
        "format": "tcga_bulk_aligned_to_world_model_v1",
        "input_dir": str(args.input_dir),
        "checkpoint": str(args.checkpoint),
        "transform": transform,
        "selection": {
            "tumor_only": True,
            "primary_only": not args.include_nonprimary,
            "exclude_ffpe": not args.include_ffpe,
            "one_file_per_patient": True,
            "limit_projects": args.limit_projects,
            "limit_patients_per_project": args.limit_patients_per_project,
        },
        "n_projects": len(manifests),
        "n_patients": int(selected["patient_id"].nunique()),
        "n_model_genes": len(model_genes),
    })


if __name__ == "__main__":
    main()
