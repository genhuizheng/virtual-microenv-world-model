#!/usr/bin/env python3
"""Rank empirical scores that clarify cross-validated patient predictions."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score

from RNA_validation.common import require_empty_output, write_json
from RNA_validation.run_empirical_scores import direction_for


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--patient-predictions", type=Path, required=True)
    p.add_argument("--empirical-scores", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def cohen_d(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2 or len(b) < 2:
        return math.nan
    pooled = math.sqrt(((len(a) - 1) * np.var(a, ddof=1) + (len(b) - 1) * np.var(b, ddof=1)) / (len(a) + len(b) - 2))
    return float((np.mean(a) - np.mean(b)) / pooled) if pooled > 0 else math.nan


def main() -> None:
    args = parse_args()
    out = require_empty_output(args.output_dir, args.overwrite)
    patients = pd.read_csv(args.patient_predictions, sep="\t", dtype=str, keep_default_na=False)
    scores = pd.read_csv(args.empirical_scores, sep="\t", compression="infer")
    probability_column = "world_finetuned_response_probability"
    if probability_column not in patients:
        raise ValueError(f"Missing {probability_column}")
    probability = pd.to_numeric(patients[probability_column], errors="coerce")
    labels = pd.to_numeric(patients["response_binary"], errors="coerce")
    base = patients[["patient_id", "biological_patient_id", "dataset_id", "source_dataset_id", "response_group", "response_binary", "fold"]].copy()
    base["model_probability"] = probability
    base["model_class"] = np.where(probability >= 0.5, 1, 0)
    base["model_correct"] = np.where(labels.notna() & probability.notna(), base["model_class"].eq(labels).astype(int), np.nan)

    id_columns = {"layer", "patient_id", "biological_patient_id", "dataset_id", "source_dataset_id", "response_binary", "cancer_code", "drug_target"}
    score_columns = [c for c in scores.columns if c not in id_columns]
    long = scores.melt(
        id_vars=[c for c in id_columns if c in scores.columns],
        value_vars=score_columns,
        var_name="score",
        value_name="raw_score",
    )
    long["raw_score"] = pd.to_numeric(long["raw_score"], errors="coerce")
    # Patient metadata is authoritative for the response label. COMPASS score
    # files may infer response_binary as int64 while patient tables retain it as
    # text, so response_binary must not be part of the identity join.
    long = long.drop(columns=["response_binary"], errors="ignore")
    merge_keys = ["patient_id", "biological_patient_id", "dataset_id", "source_dataset_id"]
    for key in merge_keys:
        long[key] = long[key].astype(str)
        base[key] = base[key].astype(str)
    long = long.merge(base, on=merge_keys, how="left", validate="many_to_one")
    long["direction"] = long["score"].map(direction_for)
    long["oriented_score"] = long["raw_score"] * long["direction"]
    mean = long.groupby(["layer", "score", "source_dataset_id"])["oriented_score"].transform("mean")
    sd = long.groupby(["layer", "score", "source_dataset_id"])["oriented_score"].transform("std").replace(0, np.nan)
    long["cohort_z_score"] = (long["oriented_score"] - mean) / sd
    long.to_csv(out / "patient_score_explanations.tsv.gz", sep="\t", index=False, compression="gzip")

    rows = []
    for (layer, score), group in long.groupby(["layer", "score"], sort=True):
        valid = group.dropna(subset=["cohort_z_score", "model_probability", "response_binary"]).copy()
        if len(valid) < 4:
            continue
        y = pd.to_numeric(valid["response_binary"], errors="coerce").astype(int).to_numpy()
        s = valid["cohort_z_score"].to_numpy(float)
        p = valid["model_probability"].to_numpy(float)
        correlation, correlation_p = spearmanr(s, p)
        responders, nonresponders = s[y == 1], s[y == 0]
        if len(responders) and len(nonresponders):
            test = mannwhitneyu(responders, nonresponders, alternative="two-sided")
            auroc = roc_auc_score(y, s)
            auprc = average_precision_score(y, s)
        else:
            test = None
            auroc = auprc = math.nan
        correct = valid.loc[valid["model_correct"].eq(1), "cohort_z_score"].to_numpy(float)
        incorrect = valid.loc[valid["model_correct"].eq(0), "cohort_z_score"].to_numpy(float)
        rows.append({
            "layer": layer, "score": score, "direction": direction_for(score), "n": len(valid),
            "score_response_auroc": auroc, "score_response_auprc": auprc,
            "responder_minus_nonresponder_cohen_d": cohen_d(responders, nonresponders),
            "response_mannwhitney_p": float(test.pvalue) if test else math.nan,
            "spearman_with_world_probability": float(correlation),
            "spearman_p": float(correlation_p),
            "correct_minus_incorrect_cohen_d": cohen_d(correct, incorrect),
            "n_model_correct": len(correct), "n_model_incorrect": len(incorrect),
        })
    ranking = pd.DataFrame(rows)
    if not ranking.empty:
        ranking["clarification_rank_score"] = (
            ranking["score_response_auroc"].fillna(0.5) - 0.5
            + 0.25 * ranking["spearman_with_world_probability"].abs().fillna(0)
            + 0.05 * ranking["responder_minus_nonresponder_cohen_d"].abs().clip(upper=5).fillna(0)
        )
        ranking = ranking.sort_values("clarification_rank_score", ascending=False)
    ranking.to_csv(out / "empirical_score_clarification_ranking.tsv", sep="\t", index=False)
    write_json(out / "run_summary.json", {
        "n_patient_score_rows": len(long),
        "n_ranked_layer_scores": len(ranking),
        "model_probability": probability_column,
        "ranking_note": "Composite rank is descriptive; primary columns remain AUROC/AUPRC, effect size, and Spearman correlation.",
    })


if __name__ == "__main__":
    main()
