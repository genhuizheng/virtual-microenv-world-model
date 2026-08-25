#!/usr/bin/env python3
"""Associate world-model predictions and empirical scores with TCGA outcomes."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from RNA_validation.common import require_empty_output, write_json
from RNA_validation.run_empirical_scores import direction_for


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--inference-dir", type=Path, required=True)
    p.add_argument("--outcomes", type=Path, required=True, help="tcga_patient_outcomes.tsv")
    p.add_argument("--empirical-scores", type=Path, default=None)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--min-patients", type=int, default=20)
    p.add_argument("--min-events", type=int, default=5)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def feature_long(inference_dir: Path, empirical_scores: Path | None) -> pd.DataFrame:
    prediction = pd.read_csv(inference_dir / "patient_prediction_features.tsv", sep="\t")
    base_features = ["predicted_delta_l2", "latent_residual_l2", "source_latent_l2"]
    long = prediction.melt(
        id_vars=["patient_id", "dataset_id"], value_vars=base_features,
        var_name="feature", value_name="value",
    )
    long["layer"] = "world_model"
    if empirical_scores:
        scores = pd.read_csv(empirical_scores, sep="\t", compression="infer")
        id_cols = [c for c in ("layer", "patient_id", "dataset_id", "source_dataset_id", "response_binary", "cancer_code", "drug_target") if c in scores.columns]
        score_cols = [c for c in scores.columns if c not in id_cols]
        score_long = scores.melt(id_vars=id_cols, value_vars=score_cols, var_name="feature", value_name="value")
        score_long = score_long[["patient_id", "dataset_id", "layer", "feature", "value"]]
        long = pd.concat([long, score_long], ignore_index=True)
    long["value"] = pd.to_numeric(long["value"], errors="coerce")
    return long.dropna(subset=["value"])


def fit_cox(frame: pd.DataFrame, duration: str, event: str, strata: bool):
    from lifelines import CoxPHFitter

    columns = [duration, event, "z"] + (["dataset_id"] if strata else [])
    data = frame[columns].dropna().copy()
    if len(data) < 3 or data[event].sum() < 1 or data["z"].std() == 0:
        return None
    model = CoxPHFitter(penalizer=0.01)
    model.fit(data, duration_col=duration, event_col=event, strata=["dataset_id"] if strata else None)
    row = model.summary.loc["z"]
    return {
        "n": len(data), "events": int(data[event].sum()),
        "hazard_ratio_per_sd": float(np.exp(row["coef"])),
        "ci_low": float(np.exp(row["coef lower 95%"])),
        "ci_high": float(np.exp(row["coef upper 95%"])),
        "p_value": float(row["p"]),
        "concordance": float(model.concordance_index_),
    }


def standardize_within_dataset(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    mean = frame.groupby("dataset_id")["value"].transform("mean")
    sd = frame.groupby("dataset_id")["value"].transform("std").replace(0, np.nan)
    frame["z"] = (frame["value"] - mean) / sd
    return frame


def main() -> None:
    args = parse_args()
    out = require_empty_output(args.output_dir, args.overwrite)
    outcomes = pd.read_csv(args.outcomes, sep="\t", dtype=str, keep_default_na=False)
    for column in ("os_time_days", "os_event", "pfs_time_days_candidate", "pfs_event_candidate", "age_at_index"):
        outcomes[column] = pd.to_numeric(outcomes[column], errors="coerce")
    features = feature_long(args.inference_dir, args.empirical_scores)
    merged = features.merge(outcomes, on=["patient_id", "dataset_id"], how="inner", validate="many_to_one")
    merged.to_csv(out / "analysis_long_table.tsv.gz", sep="\t", index=False, compression="gzip")
    cox_rows, response_rows, stratification_rows = [], [], []
    endpoints = [("OS", "os_time_days", "os_event"), ("PFS_candidate", "pfs_time_days_candidate", "pfs_event_candidate")]
    for (layer, feature), feature_data in merged.groupby(["layer", "feature"], sort=True):
        standardized = standardize_within_dataset(feature_data)
        for endpoint, duration, event in endpoints:
            pooled = standardized.dropna(subset=[duration, event, "z"])
            if len(pooled) >= args.min_patients and pooled[event].sum() >= args.min_events:
                result = fit_cox(pooled, duration, event, strata=True)
                if result:
                    cox_rows.append({"layer": layer, "feature": feature, "endpoint": endpoint,
                                     "cohort": "PAN_TCGA_STRATIFIED", "cancer_type": "PAN_TCGA", **result})
            for dataset_id, cohort in standardized.groupby("dataset_id"):
                cohort = cohort.dropna(subset=[duration, event, "z"])
                if len(cohort) < args.min_patients or cohort[event].sum() < args.min_events:
                    continue
                result = fit_cox(cohort, duration, event, strata=False)
                if result:
                    cox_rows.append({"layer": layer, "feature": feature, "endpoint": endpoint,
                                     "cohort": dataset_id, "cancer_type": str(dataset_id).removeprefix("tcga_").upper(), **result})
                median = cohort["value"].median()
                cohort = cohort.assign(group=np.where(cohort["value"] >= median, "high", "low"))
                for group, subset in cohort.groupby("group"):
                    stratification_rows.append({
                        "layer": layer, "feature": feature, "endpoint": endpoint, "cohort": dataset_id,
                        "cancer_type": str(dataset_id).removeprefix("tcga_").upper(),
                        "group": group, "n": len(subset), "events": int(subset[event].sum()),
                        "median_time_days": float(subset[duration].median()), "cutoff": float(median),
                    })
        response = standardized.loc[
            standardized["response_group"].isin(["Response", "Non-response"])
        ].dropna(subset=["z"]).copy()
        if len(response):
            y = response["response_group"].eq("Response").astype(int).to_numpy()
            direction = direction_for(feature) if layer != "world_model" else 1
            score = response["z"].to_numpy(float) * direction
            if len(np.unique(y)) == 2:
                response_rows.append({
                    "layer": layer, "feature": feature, "cohort": "PAN_TCGA",
                    "cancer_type": "PAN_TCGA",
                    "direction": direction, "n": len(y), "responders": int(y.sum()),
                    "auroc": float(roc_auc_score(y, score)), "auprc": float(average_precision_score(y, score)),
                })
            for dataset_id, cohort in response.groupby("dataset_id"):
                y = cohort["response_group"].eq("Response").astype(int).to_numpy()
                if len(cohort) < args.min_patients or len(np.unique(y)) < 2:
                    continue
                score = cohort["z"].to_numpy(float) * direction
                response_rows.append({
                    "layer": layer, "feature": feature, "cohort": dataset_id,
                    "cancer_type": str(dataset_id).removeprefix("tcga_").upper(),
                    "direction": direction, "n": len(y), "responders": int(y.sum()),
                    "auroc": float(roc_auc_score(y, score)), "auprc": float(average_precision_score(y, score)),
                })
    pd.DataFrame(cox_rows).to_csv(out / "cox_associations.tsv", sep="\t", index=False)
    pd.DataFrame(response_rows).to_csv(out / "response_associations.tsv", sep="\t", index=False)
    pd.DataFrame(stratification_rows).to_csv(out / "median_stratification_summary.tsv", sep="\t", index=False)
    write_json(out / "run_summary.json", {
        "n_joined_feature_rows": len(merged),
        "n_features": int(merged[["layer", "feature"]].drop_duplicates().shape[0]),
        "cox_results": len(cox_rows),
        "response_results": len(response_rows),
        "pfs_status": "candidate endpoint; interpret only after review of derivation flags",
        "response_status": "heterogeneous treatment outcomes; not ICI-specific",
    })


if __name__ == "__main__":
    main()
