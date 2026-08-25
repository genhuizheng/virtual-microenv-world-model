#!/usr/bin/env python3
"""Analyze pretreatment-to-predicted-post spatial biology without observed-post metrics."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import anndata as ad
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.spatial import cKDTree
from scipy.stats import mannwhitneyu, spearmanr

from RNA_validation.common import require_empty_output, write_json


MATINS_MECHANISM_GENES = {
    "STAB1": "bexmarilimab target/macrophage scavenger receptor",
    "TFRC": "macrophage activation/iron uptake",
    "CD68": "macrophage compartment",
    "APOE": "lipid-associated macrophage program",
    "APOC1": "lipid-associated macrophage program",
    "LPL": "lipid remodeling",
    "TREM2": "suppressive/lipid-associated macrophage program",
    "IL4I1": "immunoregulatory macrophage program",
    "IFNG": "IFN-gamma response",
    "CXCL9": "IFN-gamma-induced chemokine",
    "CXCL10": "IFN-gamma-induced chemokine",
    "CCL5": "T-cell recruitment/effector program",
    "CD3D": "TCR complex",
    "CD3E": "TCR complex",
    "CD3G": "TCR complex",
    "CD247": "TCR zeta chain",
    "CD28": "T-cell costimulation",
    "GZMB": "cytotoxic effector program",
    "HLA-DRA": "antigen presentation",
    "HLA-DPA1": "antigen presentation",
    "B2M": "MHC class I antigen presentation",
    "PECAM1": "vascular compartment",
    "EPCAM": "epithelial/tumor compartment",
    "KRT8": "epithelial/tumor compartment",
    "KRT18": "epithelial/tumor compartment",
}


OSIMERTINIB_MECHANISM_GENES = {
    "IFITM3": "candidate osimertinib-resistance mediator",
    "MET": "IFITM3-interacting bypass receptor",
    "EGFR": "osimertinib target pathway",
    "AKT1": "PI3K-AKT survival signaling",
    "AKT2": "PI3K-AKT survival signaling",
    "AKT3": "PI3K-AKT survival signaling",
    "PIK3CA": "PI3K-AKT survival signaling",
    "PIK3R1": "PI3K-AKT survival signaling",
    "TNF": "TME inflammatory cytokine",
    "IL6": "TME inflammatory cytokine",
    "IFNG": "TME inflammatory cytokine",
    "STAT1": "interferon-response signaling",
    "STAT3": "IL6-response signaling",
    "MKI67": "tumor proliferation",
    "TOP2A": "tumor proliferation",
    "BCL2L1": "cell-survival program",
    "MCL1": "cell-survival program",
    "CXCL9": "interferon-induced chemokine",
    "CXCL10": "interferon-induced chemokine",
    "EPCAM": "epithelial/tumor compartment",
    "KRT8": "epithelial/tumor compartment",
    "KRT18": "epithelial/tumor compartment",
}


EWS_FLI1_TARGET_GENES = {
    "NR0B1": "EWS::FLI1 GGAA-microsatellite target",
    "IL1RAP": "EWS::FLI1 GGAA-microsatellite target",
    "EZH2": "EWS::FLI1 GGAA-microsatellite target",
    "CAV1": "EWS::FLI1 GGAA-microsatellite target",
    "AKAP7": "EWS::FLI1 GGAA-microsatellite target",
    "CACNB2": "EWS::FLI1 GGAA-microsatellite target",
    "RCOR1": "EWS::FLI1 GGAA-microsatellite target",
    "FCGRT": "EWS::FLI1 GGAA-microsatellite target",
    "FEZF1": "EWS::FLI1 GGAA-microsatellite target",
    "FOCAD": "EWS::FLI1 GGAA-microsatellite target",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--prediction-dir", type=Path, required=True)
    p.add_argument("--biology-score-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--top-genes", type=int, default=30)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def response_binary(value: object) -> float:
    text = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    if text in {"response", "partial_response", "disease_control", "complete_response"}:
        return 1.0
    if text in {"non_response", "no_disease_control", "non_pr"}:
        return 0.0
    return math.nan


def add_bh_fdr(table: pd.DataFrame, p_column: str) -> pd.DataFrame:
    if p_column not in table.columns:
        table["bh_fdr"] = np.nan
        return table
    p = pd.to_numeric(table[p_column], errors="coerce").to_numpy(float)
    valid = np.flatnonzero(np.isfinite(p))
    q = np.full(len(table), np.nan)
    if len(valid):
        order = valid[np.argsort(p[valid])]
        ranked = p[order] * len(valid) / np.arange(1, len(valid) + 1)
        q[order] = np.minimum.accumulate(ranked[::-1])[::-1].clip(0, 1)
    table["bh_fdr"] = q
    return table


def dense_column(matrix, index: int) -> np.ndarray:
    column = matrix[:, index]
    if sp.issparse(column):
        column = column.toarray()
    return np.asarray(column, dtype=float).reshape(-1)


def spatial_osimertinib_rows(path: Path, sample_id: str, gse_id: str, patient_key: str) -> list[dict[str, object]]:
    spatial = ad.read_h5ad(path)
    symbols = spatial.var["gene_symbol"].fillna("").astype(str).str.upper().to_numpy()
    index = {symbol: i for i, symbol in enumerate(symbols) if symbol}
    if "IFITM3" not in index or "spatial" not in spatial.obsm or spatial.n_obs < 4:
        return []
    labels = np.repeat("all", spatial.n_obs).astype(object)
    for column in ("segment_type", "region_type"):
        if column in spatial.obs:
            labels = spatial.obs[column].fillna("all").astype(str).to_numpy()
            break
    tumor = np.array([
        any(token in str(label).lower() for token in ("tumor", "malignant", "epithelial", "cancer"))
        for label in labels
    ])
    target = tumor if tumor.sum() >= 4 else np.ones(spatial.n_obs, dtype=bool)
    scope = "tumor_annotated" if tumor.sum() >= 4 else "all_observations"
    coords = np.asarray(spatial.obsm["spatial"][:, :2], dtype=float)
    valid_coords = np.isfinite(coords).all(axis=1)
    target_index = np.flatnonzero(target & valid_coords)
    all_index = np.flatnonzero(valid_coords)
    if len(target_index) < 4 or len(all_index) < 4:
        return []
    neighbor_count = min(9, len(all_index))
    neighbor_local = cKDTree(coords[all_index]).query(coords[target_index], k=neighbor_count)[1]
    if neighbor_local.ndim == 1:
        return []
    neighbors = all_index[neighbor_local[:, 1:]]
    ifitm3_delta = dense_column(spatial.layers["predicted_delta"], index["IFITM3"])[target_index]
    rows = []
    for cytokine in ("TNF", "IL6", "IFNG"):
        if cytokine not in index:
            continue
        cytokine_delta = dense_column(spatial.layers["predicted_delta"], index[cytokine])
        cytokine_post = dense_column(spatial.X, index[cytokine])
        for quantity, values in (("neighbor_predicted_delta", cytokine_delta), ("neighbor_predicted_post", cytokine_post)):
            neighbor_mean = values[neighbors].mean(axis=1)
            if np.std(ifitm3_delta) > 0 and np.std(neighbor_mean) > 0:
                rho, pvalue = spearmanr(ifitm3_delta, neighbor_mean)
            else:
                rho, pvalue = math.nan, math.nan
            rows.append({
                "sample_id": sample_id, "gse_id": gse_id, "patient_key": patient_key,
                "target_gene": "IFITM3", "neighbor_gene": cytokine,
                "neighbor_quantity": quantity, "target_scope": scope,
                "target_observations": len(target_index), "neighbors_per_target": neighbor_count - 1,
                "spearman_rho": rho, "spearman_pvalue": pvalue,
            })
    survival_genes = [gene for gene in ("MET", "AKT1", "AKT2", "AKT3", "PIK3CA", "PIK3R1") if gene in index]
    if survival_genes:
        survival_delta = np.column_stack([
            dense_column(spatial.layers["predicted_delta"], index[gene])[target_index] for gene in survival_genes
        ]).mean(axis=1)
        if np.std(ifitm3_delta) > 0 and np.std(survival_delta) > 0:
            rho, pvalue = spearmanr(ifitm3_delta, survival_delta)
        else:
            rho, pvalue = math.nan, math.nan
        rows.append({
            "sample_id": sample_id, "gse_id": gse_id, "patient_key": patient_key,
            "target_gene": "IFITM3", "neighbor_gene": "+".join(survival_genes),
            "neighbor_quantity": "same-location_MET_PI3K_AKT_delta", "target_scope": scope,
            "target_observations": len(target_index), "neighbors_per_target": 0,
            "spearman_rho": rho, "spearman_pvalue": pvalue,
        })
    return rows


def main() -> None:
    args = parse_args()
    out = require_empty_output(args.output_dir, args.overwrite)
    (out / "figures").mkdir()
    metrics = pd.read_csv(args.prediction_dir / "sample_metrics.tsv", sep="\t")
    numeric = [
        "checkpoint_gene_coverage", "predicted_negative_fraction", "predicted_shift_l2_mean",
        "predicted_shift_l2_median", "predicted_shift_l2_q90", "predicted_shift_morans_i",
        "predicted_gene_delta_l1_mean", "predicted_gene_delta_l2", "predicted_up_genes",
        "predicted_down_genes",
    ]
    for column in numeric:
        if column not in metrics:
            metrics[column] = np.nan
        else:
            metrics[column] = pd.to_numeric(metrics[column], errors="coerce")
    for column in metrics.columns:
        if column.startswith("ewsfli1_") and column != "ewsfli1_program_reversal_predicted":
            metrics[column] = pd.to_numeric(metrics[column], errors="coerce")
    metrics["response_binary"] = metrics["response_group"].map(response_binary)
    metrics.to_csv(out / "sample_biology_metrics.tsv", sep="\t", index=False)

    for keys, filename in (
        (["gse_id", "platform", "cancer_type"], "dataset_summary.tsv"),
        (["platform"], "platform_summary.tsv"),
        (["cancer_type"], "cancer_type_summary.tsv"),
        (["response_group"], "response_summary.tsv"),
    ):
        summary = metrics.groupby(keys, dropna=False)[numeric].agg(["mean", "median", "count"])
        summary.columns = ["_".join(x) for x in summary.columns]
        summary.reset_index().to_csv(out / filename, sep="\t", index=False)

    response_metric_rows = []
    for column in numeric:
        responder = metrics.loc[metrics["response_binary"] == 1, column].dropna().to_numpy(float)
        nonresponder = metrics.loc[metrics["response_binary"] == 0, column].dropna().to_numpy(float)
        if len(responder) >= 2 and len(nonresponder) >= 2:
            statistic, pvalue = mannwhitneyu(responder, nonresponder, alternative="two-sided")
        else:
            statistic, pvalue = math.nan, math.nan
        response_metric_rows.append({
            "metric": column,
            "responders": len(responder),
            "nonresponders": len(nonresponder),
            "mean_responders": float(responder.mean()) if len(responder) else math.nan,
            "mean_nonresponders": float(nonresponder.mean()) if len(nonresponder) else math.nan,
            "responder_minus_nonresponder": float(responder.mean() - nonresponder.mean()) if len(responder) and len(nonresponder) else math.nan,
            "mannwhitney_statistic": statistic,
            "mannwhitney_pvalue": pvalue,
        })
    add_bh_fdr(pd.DataFrame(response_metric_rows), "mannwhitney_pvalue").to_csv(
        out / "sample_metric_response_associations.tsv", sep="\t", index=False
    )

    cohort_response_rows = []
    for gse_id, cohort in metrics.groupby("gse_id"):
        for column in numeric:
            responder = cohort.loc[cohort["response_binary"] == 1, column].dropna().to_numpy(float)
            nonresponder = cohort.loc[cohort["response_binary"] == 0, column].dropna().to_numpy(float)
            if len(responder) >= 2 and len(nonresponder) >= 2:
                statistic, pvalue = mannwhitneyu(responder, nonresponder, alternative="two-sided")
                cohort_response_rows.append({
                    "gse_id": gse_id,
                    "metric": column,
                    "responders": len(responder),
                    "nonresponders": len(nonresponder),
                    "responder_minus_nonresponder": float(responder.mean() - nonresponder.mean()),
                    "mannwhitney_statistic": statistic,
                    "mannwhitney_pvalue": pvalue,
                })
    add_bh_fdr(pd.DataFrame(cohort_response_rows), "mannwhitney_pvalue").to_csv(
        out / "sample_metric_response_associations_by_dataset.tsv", sep="\t", index=False
    )

    manifest = pd.read_csv(args.prediction_dir / "prediction_manifest.tsv", sep="\t", dtype=str, keep_default_na=False)
    gene_parts = []
    for row in manifest.itertuples(index=False):
        genes = pd.read_csv(args.prediction_dir / row.gene_table, sep="\t")
        measured = genes["measured_pre"].astype(str).str.lower().eq("true")
        genes = genes.loc[measured, [
            "gene_id", "gene_symbol", "pretreatment_mean", "predicted_posttreatment_mean",
            "predicted_delta", "absolute_predicted_delta",
        ]].copy()
        genes["sample_id"] = row.sample_id
        genes["gse_id"] = row.gse_id
        genes["response_group"] = row.response_group
        genes["response_binary"] = response_binary(row.response_group)
        gene_parts.append(genes)
    gene_long = pd.concat(gene_parts, ignore_index=True)
    gene_long["gene_symbol_upper"] = gene_long["gene_symbol"].fillna("").astype(str).str.upper()
    mechanism = gene_long.loc[gene_long["gene_symbol_upper"].isin(MATINS_MECHANISM_GENES)].copy()
    mechanism["mechanism"] = mechanism["gene_symbol_upper"].map(MATINS_MECHANISM_GENES)
    mechanism.to_csv(out / "matins_mechanism_gene_panel_by_sample.tsv.gz", sep="\t", index=False, compression="gzip")
    mechanism.groupby(["gene_symbol_upper", "mechanism"], as_index=False).agg(
        samples=("sample_id", "nunique"),
        pretreatment_mean=("pretreatment_mean", "mean"),
        predicted_posttreatment_mean=("predicted_posttreatment_mean", "mean"),
        mean_predicted_delta=("predicted_delta", "mean"),
        median_predicted_delta=("predicted_delta", "median"),
        fraction_predicted_up=("predicted_delta", lambda x: float((x > 0).mean())),
    ).to_csv(out / "matins_mechanism_gene_panel_summary.tsv", sep="\t", index=False)
    osimertinib = gene_long.loc[gene_long["gene_symbol_upper"].isin(OSIMERTINIB_MECHANISM_GENES)].copy()
    osimertinib["mechanism"] = osimertinib["gene_symbol_upper"].map(OSIMERTINIB_MECHANISM_GENES)
    osimertinib.to_csv(
        out / "osimertinib_resistance_gene_panel_by_sample.tsv.gz", sep="\t", index=False, compression="gzip"
    )
    osimertinib.groupby(["gene_symbol_upper", "mechanism"], as_index=False).agg(
        samples=("sample_id", "nunique"),
        pretreatment_mean=("pretreatment_mean", "mean"),
        predicted_posttreatment_mean=("predicted_posttreatment_mean", "mean"),
        mean_predicted_delta=("predicted_delta", "mean"),
        median_predicted_delta=("predicted_delta", "median"),
        fraction_predicted_up=("predicted_delta", lambda x: float((x > 0).mean())),
    ).to_csv(out / "osimertinib_resistance_gene_panel_summary.tsv", sep="\t", index=False)
    ews_fli1 = gene_long.loc[
        gene_long["gene_symbol_upper"].isin(EWS_FLI1_TARGET_GENES) & gene_long["gse_id"].eq("GSE322640")
    ].copy()
    ews_fli1["mechanism"] = ews_fli1["gene_symbol_upper"].map(EWS_FLI1_TARGET_GENES)
    ews_fli1.to_csv(out / "ewsfli1_target_panel_by_sample.tsv.gz", sep="\t", index=False, compression="gzip")
    ews_fli1.groupby(["gene_symbol_upper", "mechanism"], as_index=False).agg(
        samples=("sample_id", "nunique"),
        pretreatment_mean=("pretreatment_mean", "mean"),
        predicted_posttreatment_mean=("predicted_posttreatment_mean", "mean"),
        mean_predicted_delta=("predicted_delta", "mean"),
        median_predicted_delta=("predicted_delta", "median"),
        fraction_predicted_down=("predicted_delta", lambda x: float((x < 0).mean())),
    ).to_csv(out / "ewsfli1_target_panel_summary.tsv", sep="\t", index=False)
    ews_metric_columns = [column for column in metrics.columns if column.startswith("ewsfli1_")]
    ewing_metrics = metrics.loc[metrics["gse_id"].eq("GSE322640")].copy()
    ewing_metrics[[
        "sample_id", "gse_id", "patient_key", "cancer_type", *ews_metric_columns
    ]].to_csv(
        out / "ewsfli1_module_scores_by_sample.tsv", sep="\t", index=False
    )
    if not ewing_metrics.empty and "ewsfli1_module_predicted_delta_mean" in ewing_metrics:
        pd.DataFrame([{
            "patients": int(ewing_metrics["patient_key"].nunique()),
            "mean_predicted_score_change": float(ewing_metrics["ewsfli1_module_predicted_delta_mean"].mean()),
            "median_predicted_score_change": float(ewing_metrics["ewsfli1_module_predicted_delta_mean"].median()),
            "patients_with_predicted_reversal": int(
                ewing_metrics["ewsfli1_program_reversal_predicted"].astype(str).str.lower().eq("true").sum()
            ),
            "response_analysis_available": False,
            "reason": "GSE322640 standardized response labels are NA",
        }]).to_csv(out / "ewsfli1_module_summary.tsv", sep="\t", index=False)

    grouped = gene_long.groupby(["gene_id", "gene_symbol"], dropna=False)
    gene_summary = grouped["predicted_delta"].agg(["count", "mean", "median", "std"]).reset_index()
    gene_summary = gene_summary.rename(columns={"count": "samples", "mean": "mean_predicted_delta", "median": "median_predicted_delta", "std": "sd_predicted_delta"})
    direction = grouped["predicted_delta"].agg(
        fraction_predicted_up=lambda x: float((x > 0).mean()),
        fraction_predicted_down=lambda x: float((x < 0).mean()),
        mean_absolute_predicted_delta=lambda x: float(np.abs(x).mean()),
    ).reset_index()
    gene_summary = gene_summary.merge(direction, on=["gene_id", "gene_symbol"], how="left")
    gene_summary["direction_consistency"] = gene_summary[["fraction_predicted_up", "fraction_predicted_down"]].max(axis=1)
    gene_summary.sort_values(["direction_consistency", "mean_absolute_predicted_delta"], ascending=[False, False]).to_csv(
        out / "predicted_gene_change_consistency.tsv", sep="\t", index=False
    )

    association_rows = []
    for (gene_id, gene_symbol), group in gene_long.groupby(["gene_id", "gene_symbol"], dropna=False):
        responder = group.loc[group["response_binary"] == 1, "predicted_delta"].to_numpy(float)
        nonresponder = group.loc[group["response_binary"] == 0, "predicted_delta"].to_numpy(float)
        if len(responder) >= 2 and len(nonresponder) >= 2:
            statistic, pvalue = mannwhitneyu(responder, nonresponder, alternative="two-sided")
        else:
            statistic, pvalue = math.nan, math.nan
        association_rows.append({
            "gene_id": gene_id,
            "gene_symbol": gene_symbol,
            "responders": len(responder),
            "nonresponders": len(nonresponder),
            "mean_delta_responders": float(responder.mean()) if len(responder) else math.nan,
            "mean_delta_nonresponders": float(nonresponder.mean()) if len(nonresponder) else math.nan,
            "responder_minus_nonresponder": float(responder.mean() - nonresponder.mean()) if len(responder) and len(nonresponder) else math.nan,
            "mannwhitney_statistic": statistic,
            "mannwhitney_pvalue": pvalue,
        })
    add_bh_fdr(pd.DataFrame(association_rows), "mannwhitney_pvalue").to_csv(
        out / "predicted_gene_response_associations.tsv", sep="\t", index=False
    )

    dataset_gene_rows = []
    for (gse_id, gene_id, gene_symbol), group in gene_long.groupby(
        ["gse_id", "gene_id", "gene_symbol"], dropna=False
    ):
        responder = group.loc[group["response_binary"] == 1, "predicted_delta"].to_numpy(float)
        nonresponder = group.loc[group["response_binary"] == 0, "predicted_delta"].to_numpy(float)
        if len(responder) >= 2 and len(nonresponder) >= 2:
            statistic, pvalue = mannwhitneyu(responder, nonresponder, alternative="two-sided")
            dataset_gene_rows.append({
                "gse_id": gse_id, "gene_id": gene_id, "gene_symbol": gene_symbol,
                "responders": len(responder), "nonresponders": len(nonresponder),
                "mean_delta_responders": float(responder.mean()),
                "mean_delta_nonresponders": float(nonresponder.mean()),
                "responder_minus_nonresponder": float(responder.mean() - nonresponder.mean()),
                "mannwhitney_statistic": statistic, "mannwhitney_pvalue": pvalue,
            })
    add_bh_fdr(pd.DataFrame(dataset_gene_rows), "mannwhitney_pvalue").to_csv(
        out / "predicted_gene_response_associations_by_dataset.tsv.gz",
        sep="\t", index=False, compression="gzip",
    )

    hotspots = pd.read_csv(args.prediction_dir / "spatial_gene_hotspots.tsv", sep="\t")
    hotspot_summary = hotspots.groupby(["gene_id", "gene_symbol"], dropna=False).agg(
        samples=("sample_id", "nunique"),
        mean_predicted_delta=("mean_predicted_delta", "mean"),
        median_spatial_delta_morans_i=("spatial_delta_morans_i", "median"),
        mean_spatial_delta_morans_i=("spatial_delta_morans_i", "mean"),
    ).reset_index()
    hotspot_summary.sort_values("median_spatial_delta_morans_i", ascending=False).to_csv(
        out / "spatial_gene_hotspot_summary.tsv", sep="\t", index=False
    )

    spatial_resistance_rows = []
    for row in manifest.itertuples(index=False):
        if row.gse_id in {"GSE288758", "GSE301973"}:
            spatial_resistance_rows.extend(spatial_osimertinib_rows(
                args.prediction_dir / row.predicted_h5ad, row.sample_id, row.gse_id, row.patient_key
            ))
    spatial_resistance = add_bh_fdr(pd.DataFrame(spatial_resistance_rows), "spearman_pvalue")
    spatial_resistance.to_csv(out / "osimertinib_spatial_cytokine_associations.tsv", sep="\t", index=False)

    segment_manifest_path = args.prediction_dir / "segment_prediction_manifest.tsv"
    segment_count = 0
    if segment_manifest_path.exists():
        segment_manifest = pd.read_csv(segment_manifest_path, sep="\t", dtype=str, keep_default_na=False)
        segment_parts = []
        for row in segment_manifest.itertuples(index=False):
            genes = pd.read_csv(args.prediction_dir / row.segment_gene_table, sep="\t")
            measured = genes["measured_pre"].astype(str).str.lower().eq("true")
            genes = genes.loc[measured, [
                "gene_id", "gene_symbol", "pretreatment_mean",
                "predicted_posttreatment_mean", "predicted_delta",
            ]].copy()
            genes["sample_id"] = row.sample_id
            genes["patient_key"] = row.patient_key
            genes["gse_id"] = row.gse_id
            genes["response_group"] = row.response_group
            genes["segment_type"] = row.segment_type
            segment_parts.append(genes)
        if segment_parts:
            segment_long = pd.concat(segment_parts, ignore_index=True)
            segment_count = int(segment_long["segment_type"].nunique())
            segment_long.groupby(["segment_type", "gene_id", "gene_symbol"], dropna=False).agg(
                samples=("sample_id", "nunique"),
                patients=("patient_key", "nunique"),
                pretreatment_mean=("pretreatment_mean", "mean"),
                predicted_posttreatment_mean=("predicted_posttreatment_mean", "mean"),
                mean_predicted_delta=("predicted_delta", "mean"),
                median_predicted_delta=("predicted_delta", "median"),
            ).reset_index().to_csv(out / "predicted_gene_changes_by_segment.tsv.gz", sep="\t", index=False, compression="gzip")
            segment_long["gene_symbol_upper"] = segment_long["gene_symbol"].fillna("").astype(str).str.upper()
            segment_mechanism = segment_long.loc[
                segment_long["gene_symbol_upper"].isin(MATINS_MECHANISM_GENES)
            ].copy()
            segment_mechanism["mechanism"] = segment_mechanism["gene_symbol_upper"].map(MATINS_MECHANISM_GENES)
            segment_mechanism.groupby(["segment_type", "gene_symbol_upper", "mechanism"], as_index=False).agg(
                samples=("sample_id", "nunique"),
                patients=("patient_key", "nunique"),
                pretreatment_mean=("pretreatment_mean", "mean"),
                predicted_posttreatment_mean=("predicted_posttreatment_mean", "mean"),
                mean_predicted_delta=("predicted_delta", "mean"),
            ).to_csv(out / "matins_mechanism_gene_panel_by_segment.tsv", sep="\t", index=False)
            segment_osimertinib = segment_long.loc[
                segment_long["gene_symbol_upper"].isin(OSIMERTINIB_MECHANISM_GENES)
            ].copy()
            segment_osimertinib["mechanism"] = segment_osimertinib["gene_symbol_upper"].map(
                OSIMERTINIB_MECHANISM_GENES
            )
            segment_osimertinib.groupby(
                ["segment_type", "gene_symbol_upper", "mechanism"], as_index=False
            ).agg(
                samples=("sample_id", "nunique"),
                patients=("patient_key", "nunique"),
                pretreatment_mean=("pretreatment_mean", "mean"),
                predicted_posttreatment_mean=("predicted_posttreatment_mean", "mean"),
                mean_predicted_delta=("predicted_delta", "mean"),
            ).to_csv(out / "osimertinib_resistance_gene_panel_by_segment.tsv", sep="\t", index=False)
            segment_ews_fli1 = segment_long.loc[
                segment_long["gene_symbol_upper"].isin(EWS_FLI1_TARGET_GENES)
                & segment_long["gse_id"].eq("GSE322640")
            ].copy()
            segment_ews_fli1["mechanism"] = segment_ews_fli1["gene_symbol_upper"].map(EWS_FLI1_TARGET_GENES)
            segment_ews_fli1.groupby(
                ["segment_type", "gene_symbol_upper", "mechanism"], as_index=False
            ).agg(
                samples=("sample_id", "nunique"),
                patients=("patient_key", "nunique"),
                pretreatment_mean=("pretreatment_mean", "mean"),
                predicted_posttreatment_mean=("predicted_posttreatment_mean", "mean"),
                mean_predicted_delta=("predicted_delta", "mean"),
                fraction_predicted_down=("predicted_delta", lambda x: float((x < 0).mean())),
            ).to_csv(out / "ewsfli1_target_panel_by_segment.tsv", sep="\t", index=False)

    top = gene_summary.nlargest(args.top_genes, "mean_absolute_predicted_delta").sort_values("mean_predicted_delta")
    fig, ax = plt.subplots(figsize=(8, max(5, len(top) * 0.22)))
    colors = np.where(top["mean_predicted_delta"] >= 0, "#C44E52", "#4C72B0")
    ax.barh(top["gene_symbol"].fillna(top["gene_id"]), top["mean_predicted_delta"], color=colors)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("Predicted posttreatment - pretreatment expression")
    ax.set_title("Largest model-predicted biological changes")
    fig.tight_layout()
    fig.savefig(out / "figures" / "top_predicted_gene_changes.png", dpi=180)
    plt.close(fig)

    score_delta = pd.read_csv(args.biology_score_dir / "empirical_score_deltas.tsv.gz", sep="\t")
    score_columns = [column for column in score_delta.columns if column.endswith("__delta")]
    score_summary = score_delta[score_columns].agg(["mean", "median", "std"]).T.reset_index().rename(columns={"index": "score_delta"})
    score_summary.to_csv(out / "empirical_score_delta_summary.tsv", sep="\t", index=False)

    write_json(out / "analysis_summary.json", {
        "baseline_samples": int(len(metrics)),
        "patients": int(metrics["patient_key"].nunique()),
        "datasets": int(metrics["gse_id"].nunique()),
        "median_predicted_latent_shift": float(metrics["predicted_shift_l2_median"].median()),
        "median_spatial_shift_morans_i": float(metrics["predicted_shift_morans_i"].median()),
        "genes_analyzed": int(gene_summary["gene_id"].nunique()),
        "empirical_scores_analyzed": len(score_columns),
        "segment_types_analyzed": segment_count,
        "osimertinib_spatial_associations": int(len(spatial_resistance)),
        "gse322640_baseline_patients": int(ewing_metrics["patient_key"].nunique()),
        "paper_examples": [
            "MATINS/GSE240138 compartment-aware macrophage remodeling",
            "GSE288758 osimertinib IFITM3-MET-AKT resistance biology",
            "SARC037/GSE322640 predicted EWS::FLI1 program reversal",
        ],
        "measured_posttreatment_used": False,
        "analysis_contrast": "pretreatment versus model-predicted posttreatment",
        "warning": "Predicted changes are hypotheses from a generic transition model, not measured treatment effects or causal estimates.",
    })
    print(f"biology_analysis_samples={len(metrics)} output={out}")


if __name__ == "__main__":
    main()
