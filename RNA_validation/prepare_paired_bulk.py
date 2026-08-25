#!/usr/bin/env python3
"""Align the standardized paired bulk RNA-seq cohort to a world-model checkpoint."""

from __future__ import annotations

import argparse
import gzip
from pathlib import Path

import numpy as np
import pandas as pd

from RNA_validation.common import (
    GeneAligner,
    checkpoint_gene_table,
    clean,
    load_checkpoint,
    read_tsv,
    require_empty_output,
    transform_counts,
    write_json,
)


CANCER_TO_TCGA = {
    "MELANOMA": "SKCM", "GLIOBLASTOMA": "GBM", "COLORECTAL CANCER": "COAD",
    "BREAST CANCER": "BRCA", "HER2-RECEPTOR NEGATIVE BREAST CANCER": "BRCA",
    "HER2-RECEPTOR POSITIVE BREAST CANCER": "BRCA", "ESTROGEN-RECEPTOR POSITIVE BREAST CANCER": "BRCA",
    "HEAD AND NECK SQUAMOUS CELL CARCINOMA": "HNSC", "OVARIAN CANCER": "OV",
    "OVARIAN SEROUS CARCINOMA": "OV", "PROSTATE CANCER": "PRAD", "ESOPHAGEAL CANCER": "ESCA",
}


def drug_target(regimen: str) -> str:
    value = regimen.upper()
    if any(x in value for x in ("NIVOLUMAB", "PEMBROLIZUMAB", "CEMIPLIMAB")):
        return "PD1"
    if any(x in value for x in ("ATEZOLIZUMAB", "DURVALUMAB", "AVELUMAB")):
        return "PDL1"
    if any(x in value for x in ("IPILIMUMAB", "TREMELIMUMAB")):
        return "CTLA4"
    return "NA"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input-dir", type=Path, required=True, help="standardized_rnaseq directory")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--model-genes", type=Path, default=None)
    p.add_argument("--ranked-reference", type=Path, default=None)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--transform", choices=["auto", "none", "cpm", "log1p_cpm", "log1p_10k"], default="auto")
    p.add_argument("--limit-datasets", type=int, default=None, help="Deterministic smoke-test limit after sorting datasets")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.limit_datasets is not None and args.limit_datasets < 1:
        raise ValueError("--limit-datasets must be positive")
    out = require_empty_output(args.output_dir, args.overwrite)
    checkpoint = load_checkpoint(args.checkpoint)
    transform = (
        str(checkpoint.get("args", {}).get("expression_transform", "none"))
        if args.transform == "auto" else args.transform
    )
    model_genes = checkpoint_gene_table(checkpoint, args.model_genes)
    aligner = GeneAligner(model_genes, args.ranked_reference)

    studies = read_tsv(args.input_dir / "studies.tsv")
    patients = read_tsv(args.input_dir / "patients.tsv")
    samples = read_tsv(args.input_dir / "samples.tsv")
    pairs = read_tsv(args.input_dir / "patient_pairs.tsv")
    studies_i = studies.set_index("dataset_id", drop=False)
    patients_i = patients.set_index("patient_id", drop=False)
    samples_i = samples.set_index("sample_id", drop=False)

    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    meta_rows: list[dict[str, str]] = []
    alignment_summaries = []
    detailed_path = out / "gene_alignment_by_dataset.tsv.gz"
    wrote_detail = False

    matrix_paths = sorted((args.input_dir / "counts").glob("*_counts.tsv"))
    if args.limit_datasets is not None:
        matrix_paths = matrix_paths[:args.limit_datasets]
    for matrix_path in matrix_paths:
        dataset_id = matrix_path.name.removesuffix("_counts.tsv")
        matrix = pd.read_csv(matrix_path, sep="\t", dtype={"gene_id": str, "gene_symbol": str})
        if list(matrix.columns[:2]) != ["gene_id", "gene_symbol"]:
            raise ValueError(f"Unexpected gene columns in {matrix_path}")
        sample_columns = list(matrix.columns[2:])
        raw = matrix.iloc[:, 2:].to_numpy(dtype=np.float32)
        if np.any(raw < 0) or not np.all(np.isfinite(raw)):
            raise ValueError(f"Invalid counts in {matrix_path}")
        library_sizes = raw.sum(axis=0, dtype=np.float64)
        alignment = aligner.resolve(matrix["gene_id"], matrix["gene_symbol"])
        aligned_raw = aligner.align_gene_by_sample(raw, alignment)
        aligned = transform_counts(aligned_raw, library_sizes, transform)
        col_to_i = {column: i for i, column in enumerate(sample_columns)}

        detail = alignment.report.copy()
        detail.insert(0, "dataset_id", dataset_id)
        detail.to_csv(
            detailed_path,
            sep="\t",
            index=False,
            mode="at" if wrote_detail else "wt",
            header=not wrote_detail,
            compression="gzip",
        )
        wrote_detail = True
        method_counts = detail["mapping_method"].value_counts().to_dict()
        alignment_summaries.append({
            "dataset_id": dataset_id,
            "source_genes": len(detail),
            "mapped_source_rows": int((alignment.source_to_model >= 0).sum()),
            "unique_model_genes_observed": int(len(set(alignment.source_to_model[alignment.source_to_model >= 0]))),
            "model_genes": len(model_genes),
            "model_gene_coverage": float(len(set(alignment.source_to_model[alignment.source_to_model >= 0])) / len(model_genes)),
            **{f"mapping_{key}": int(value) for key, value in method_counts.items()},
        })

        for pair in pairs.loc[pairs["dataset_id"].eq(dataset_id)].itertuples(index=False):
            if pair.pre_sample_id not in col_to_i or pair.post_sample_id not in col_to_i:
                raise ValueError(f"Pair columns missing for {pair.patient_id}")
            patient = patients_i.loc[pair.patient_id]
            post = samples_i.loc[pair.post_sample_id]
            study = studies_i.loc[dataset_id]
            response = clean(patient["response_group"])
            binary = "1" if response == "Response" else "0" if response == "Non-response" else "NA"
            source_patient = clean(patient["source_patient_id"])
            accession = clean(study["source_dataset_id"])
            biological_patient = f"{accession}__{source_patient}"
            regimen = clean(post["therapeutic_regimen"]) or "NA"
            cancer = clean(patient["cancer_subtype_standard"]) or "NA"
            xs.append(aligned[col_to_i[pair.pre_sample_id]])
            ys.append(aligned[col_to_i[pair.post_sample_id]])
            meta_rows.append({
                "row_index": str(len(meta_rows)),
                "pair_id": pair.pair_id,
                "patient_id": pair.patient_id,
                "biological_patient_id": biological_patient,
                "source_patient_id": source_patient,
                "dataset_id": dataset_id,
                "source_dataset_id": accession,
                "pre_sample_id": pair.pre_sample_id,
                "post_sample_id": pair.post_sample_id,
                "response_group": response or "NA",
                "response_binary": binary,
                "original_response_status": clean(patient["original_response_status"]) or "NA",
                "cancer_subtype": cancer,
                "cancer_code": CANCER_TO_TCGA.get(cancer.upper(), "NA"),
                "therapeutic_regimen": regimen,
                "drug_target": drug_target(regimen),
            })

    if not xs:
        raise ValueError("No paired count matrices were prepared")
    x = np.stack(xs).astype(np.float32)
    y = np.stack(ys).astype(np.float32)
    metadata = pd.DataFrame(meta_rows)
    metadata.to_csv(out / "patients.tsv", sep="\t", index=False)
    pd.DataFrame(alignment_summaries).to_csv(out / "gene_alignment_summary.tsv", sep="\t", index=False)
    model_genes.to_csv(out / "model_genes.tsv", sep="\t", index=False)
    np.savez_compressed(out / "paired_expression.npz", x_pre=x, y_post=y)
    write_json(out / "manifest.json", {
        "format": "paired_bulk_aligned_to_world_model_v1",
        "input_dir": str(args.input_dir),
        "checkpoint": str(args.checkpoint),
        "ranked_reference": str(args.ranked_reference) if args.ranked_reference else None,
        "transform": transform,
        "limit_datasets": args.limit_datasets,
        "n_pair_records": len(metadata),
        "n_biological_patients": int(metadata["biological_patient_id"].nunique()),
        "n_model_genes": len(model_genes),
        "n_labeled": int(metadata["response_binary"].isin(["0", "1"]).sum()),
        "warning": "Bulk-to-single-cell transfer is distribution-shifted; use the same declared transform for every method.",
    })
    print(f"prepared_pairs={len(metadata)} model_genes={len(model_genes)} output={out}")


if __name__ == "__main__":
    main()
