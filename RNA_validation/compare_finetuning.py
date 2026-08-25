#!/usr/bin/env python3
"""Compare pretrained and fine-tuned world-model results on identical held-out patients."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from RNA_validation.finetune_paired_cv import summarize_finetuning


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--cv-dir", type=Path, required=True)
    p.add_argument("--prepared-dir", type=Path, required=True)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    metrics = pd.read_csv(args.cv_dir / "expression_metrics_by_patient.tsv", sep="\t")
    metadata = pd.read_csv(
        args.prepared_dir / "patients.tsv", sep="\t", dtype=str, keep_default_na=False
    )
    by_record, summary = summarize_finetuning(metrics, metadata, args.seed)
    by_record.to_csv(args.cv_dir / "finetuning_comparison_by_record.tsv", sep="\t", index=False)
    summary.to_csv(args.cv_dir / "finetuning_comparison_summary.tsv", sep="\t", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
