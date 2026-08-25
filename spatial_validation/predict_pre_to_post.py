#!/usr/bin/env python3
"""Predict a biological post-treatment state from each baseline spatial sample."""

from __future__ import annotations

import argparse
import gzip
import re
from pathlib import Path

import anndata as ad
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from RNA_validation.common import (
    GeneAligner,
    build_world_model_from_checkpoint,
    checkpoint_gene_table,
    choose_device,
    clean,
    load_checkpoint,
    require_empty_output,
    write_json,
)
from spatial_validation.common import aligned_batches, alignment_matrix, coordinates, morans_i


SPATIAL_MECHANISM_GENES = {
    # MATINS macrophage-remodeling example.
    "STAB1", "TFRC", "CD68", "APOE", "APOC1", "LPL", "TREM2", "IL4I1",
    "IFNG", "CXCL9", "CXCL10", "CCL5", "CD3D", "CD3E", "CD3G", "CD247",
    "CD28", "GZMB", "HLA-DRA", "HLA-DPA1", "B2M", "PECAM1", "EPCAM",
    "KRT8", "KRT18",
    # Osimertinib-resistance/IFITM3-MET-AKT example.
    "IFITM3", "MET", "EGFR", "AKT1", "AKT2", "AKT3", "PIK3CA", "PIK3R1",
    "TNF", "IL6", "STAT1", "STAT3", "MKI67", "TOP2A", "BCL2L1", "MCL1",
    # SARC037 composite EWS::FLI1-induced target panel.
    "NR0B1", "IL1RAP", "EZH2", "CAV1", "AKAP7", "CACNB2", "RCOR1", "FCGRT",
    "FEZF1", "FOCAD",
}

EWS_FLI1_TARGETS = {
    "NR0B1", "IL1RAP", "EZH2", "CAV1", "AKAP7", "CACNB2", "RCOR1", "FCGRT", "FEZF1", "FOCAD",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest-dir", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--model-genes", type=Path)
    p.add_argument("--ranked-reference", type=Path)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--flow-steps", type=int, default=8)
    p.add_argument("--max-pre-observations", type=int, default=-1)
    p.add_argument("--spatial-top-genes", type=int, default=50)
    p.add_argument("--device", default="auto")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def safe_label(value: object) -> str:
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", clean(value)).strip("_.")
    return text or "all"


def segment_labels(adata, n: int) -> np.ndarray:
    for column in ("segment_type", "region_type"):
        if column in adata.obs:
            values = adata.obs[column].iloc[:n].map(lambda x: clean(x) or "all").astype(str).to_numpy()
            if np.any(values != "all"):
                return values
    return np.repeat("all", n)


def expression_matched_controls(
    pre_mean: np.ndarray, measured: np.ndarray, signature_index: np.ndarray,
    bins: int = 24, controls_per_gene: int = 100,
) -> np.ndarray:
    """Deterministic AddModuleScore-like control genes matched on baseline abundance."""
    eligible = np.flatnonzero(measured & np.isfinite(pre_mean))
    eligible = eligible[np.argsort(pre_mean[eligible], kind="stable")]
    bin_id = np.full(len(pre_mean), -1, dtype=int)
    if len(eligible):
        bin_id[eligible] = np.minimum((np.arange(len(eligible)) * bins) // len(eligible), bins - 1)
    signature_set = set(signature_index.tolist())
    selected = []
    for gene_index in signature_index:
        candidates = eligible[bin_id[eligible] == bin_id[gene_index]]
        candidates = np.asarray([x for x in candidates if int(x) not in signature_set], dtype=int)
        if len(candidates) > controls_per_gene:
            take = np.linspace(0, len(candidates) - 1, controls_per_gene, dtype=int)
            candidates = candidates[take]
        selected.extend(candidates.tolist())
    return np.unique(np.asarray(selected, dtype=int))


def predict_batch(model, x: np.ndarray, device: torch.device, flow_steps: int):
    xb = torch.from_numpy(x).to(device)
    z_source = model.encode(xb)
    delta = torch.ones(len(xb), device=device, dtype=z_source.dtype)
    if int(model.config.level) == 6:
        z_pred = model.integrate_cellflow(z_source, delta, n_steps=flow_steps, method="euler")
    else:
        z_pred = model.transition(z_source, delta)["z_pred"]
    return model.decode(z_pred).cpu().numpy(), z_source.cpu().numpy(), z_pred.cpu().numpy()


def plot_shift(path: Path, coords: np.ndarray, shift: np.ndarray, title: str) -> None:
    valid = np.isfinite(coords).all(axis=1) & np.isfinite(shift)
    idx = np.flatnonzero(valid)
    if len(idx) > 100_000:
        idx = np.random.default_rng(42).choice(idx, 100_000, replace=False)
    fig, ax = plt.subplots(figsize=(7, 6))
    scatter = ax.scatter(coords[idx, 0], coords[idx, 1], c=shift[idx], s=2, cmap="magma", linewidths=0)
    ax.set_title(title)
    ax.set_xlabel("spatial x")
    ax.set_ylabel("spatial y")
    ax.set_aspect("equal", adjustable="datalim")
    fig.colorbar(scatter, ax=ax, label="predicted latent transition L2")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    out = require_empty_output(args.output_dir, args.overwrite)
    for name in ("latent", "gene_tables", "segment_gene_tables", "alignment", "figures", "predicted_h5ad"):
        (out / name).mkdir()
    checkpoint = load_checkpoint(args.checkpoint)
    model_genes = checkpoint_gene_table(checkpoint, args.model_genes)
    transform = str(checkpoint.get("args", {}).get("expression_transform", checkpoint.get("expression_transform", "none")))
    if transform not in {"none", "log1p_10k"}:
        raise ValueError(f"Unsupported checkpoint expression transform: {transform}")
    aligner = GeneAligner(model_genes, args.ranked_reference)
    device = choose_device(args.device)
    model = build_world_model_from_checkpoint(checkpoint, device).eval()
    baselines = pd.read_csv(args.manifest_dir / "spatial_baselines.tsv", sep="\t", dtype=str, keep_default_na=False)
    metric_rows, result_rows, hotspot_rows, segment_rows = [], [], [], []

    with torch.no_grad():
        for sample in baselines.itertuples(index=False):
            print(f"predicting {sample.sample_id}", flush=True)
            pre = ad.read_h5ad(sample.pre_file, backed="r")
            try:
                mapping, measured, report = alignment_matrix(aligner, pre)
                if measured.sum() < 100:
                    raise ValueError(f"Only {measured.sum()} measured checkpoint genes for {sample.sample_id}")
                n_pre = pre.n_obs if args.max_pre_observations < 0 else min(pre.n_obs, args.max_pre_observations)
                labels = segment_labels(pre, n_pre)
                pre_sum = np.zeros(len(model_genes), dtype=np.float64)
                pred_sum = np.zeros(len(model_genes), dtype=np.float64)
                segment_pre_sum: dict[str, np.ndarray] = {}
                segment_pred_sum: dict[str, np.ndarray] = {}
                segment_counts: dict[str, int] = {}
                z_source_parts, z_pred_parts, shift_parts = [], [], []
                negative = 0
                predicted_values = 0
                for start, stop, expression in aligned_batches(pre, mapping, transform, args.batch_size, args.max_pre_observations):
                    pred, z_source_batch, z_pred_batch = predict_batch(model, expression, device, args.flow_steps)
                    pre_sum += expression.sum(axis=0, dtype=np.float64)
                    pred_sum += pred.sum(axis=0, dtype=np.float64)
                    negative += int((pred < 0).sum())
                    predicted_values += pred.size
                    z_source_parts.append(z_source_batch.astype(np.float16))
                    z_pred_parts.append(z_pred_batch.astype(np.float16))
                    shift_parts.append(np.linalg.norm(z_pred_batch - z_source_batch, axis=1).astype(np.float32))
                    batch_labels = labels[start:stop]
                    for label in np.unique(batch_labels):
                        select = batch_labels == label
                        segment_pre_sum.setdefault(label, np.zeros(len(model_genes), dtype=np.float64))
                        segment_pred_sum.setdefault(label, np.zeros(len(model_genes), dtype=np.float64))
                        segment_counts[label] = segment_counts.get(label, 0) + int(select.sum())
                        segment_pre_sum[label] += expression[select].sum(axis=0, dtype=np.float64)
                        segment_pred_sum[label] += pred[select].sum(axis=0, dtype=np.float64)
                pre_mean = (pre_sum / max(n_pre, 1)).astype(np.float32)
                pred_mean = (pred_sum / max(n_pre, 1)).astype(np.float32)
                predicted_delta = pred_mean - pre_mean
                z_source = np.concatenate(z_source_parts).astype(np.float32)
                z_pred = np.concatenate(z_pred_parts).astype(np.float32)
                shift = np.concatenate(shift_parts)
                coords = coordinates(pre, n_pre)
                measured_delta = predicted_delta[measured]

                metric_rows.append({
                    "sample_id": sample.sample_id,
                    "gse_id": sample.gse_id,
                    "patient_key": sample.patient_key,
                    "patient_id": sample.patient_id,
                    "platform": sample.platform,
                    "cancer_type": sample.cancer_type,
                    "response_group": sample.response_group,
                    "observations_used": n_pre,
                    "measured_checkpoint_genes": int(measured.sum()),
                    "checkpoint_gene_coverage": float(measured.mean()),
                    "predicted_negative_fraction": negative / max(predicted_values, 1),
                    "predicted_shift_l2_mean": float(shift.mean()),
                    "predicted_shift_l2_median": float(np.median(shift)),
                    "predicted_shift_l2_q90": float(np.quantile(shift, 0.9)),
                    "predicted_shift_morans_i": morans_i(shift, coords),
                    "predicted_gene_delta_l1_mean": float(np.mean(np.abs(measured_delta))),
                    "predicted_gene_delta_l2": float(np.linalg.norm(measured_delta)),
                    "predicted_up_genes": int((measured_delta > 0).sum()),
                    "predicted_down_genes": int((measured_delta < 0).sum()),
                })

                np.savez_compressed(
                    out / "latent" / f"{sample.sample_id}.npz",
                    spatial=coords,
                    z_source=z_source.astype(np.float16),
                    z_pred=z_pred.astype(np.float16),
                    predicted_shift_l2=shift,
                )
                gene_table = model_genes.copy()
                gene_table["measured_pre"] = measured
                gene_table["pretreatment_mean"] = pre_mean
                gene_table["predicted_posttreatment_mean"] = pred_mean
                gene_table["predicted_delta"] = predicted_delta
                gene_table["absolute_predicted_delta"] = np.abs(predicted_delta)
                gene_path = out / "gene_tables" / f"{sample.sample_id}.tsv.gz"
                gene_table.to_csv(gene_path, sep="\t", index=False, compression="gzip")

                for label in sorted(segment_counts):
                    count = segment_counts[label]
                    segment_table = model_genes.copy()
                    segment_table["measured_pre"] = measured
                    segment_table["pretreatment_mean"] = (segment_pre_sum[label] / count).astype(np.float32)
                    segment_table["predicted_posttreatment_mean"] = (segment_pred_sum[label] / count).astype(np.float32)
                    segment_table["predicted_delta"] = (
                        segment_table["predicted_posttreatment_mean"] - segment_table["pretreatment_mean"]
                    )
                    segment_path = out / "segment_gene_tables" / f"{sample.sample_id}__{safe_label(label)}.tsv.gz"
                    segment_table.to_csv(segment_path, sep="\t", index=False, compression="gzip")
                    segment_rows.append({
                        "sample_id": sample.sample_id,
                        "gse_id": sample.gse_id,
                        "patient_key": sample.patient_key,
                        "patient_id": sample.patient_id,
                        "platform": sample.platform,
                        "cancer_type": sample.cancer_type,
                        "response_group": sample.response_group,
                        "segment_type": label,
                        "observations": count,
                        "segment_gene_table": str(segment_path.relative_to(out)),
                    })

                eligible_index = np.flatnonzero(measured)
                top_count = min(args.spatial_top_genes, len(eligible_index))
                top_index = eligible_index[np.argsort(np.abs(predicted_delta[eligible_index]))[-top_count:]]
                symbols_upper = model_genes["gene_symbol"].fillna("").astype(str).str.upper().to_numpy()
                mechanism_index = np.flatnonzero(measured & np.isin(symbols_upper, list(SPATIAL_MECHANISM_GENES)))
                top_index = np.unique(np.concatenate([top_index, mechanism_index]))
                ews_fli1_index = np.flatnonzero(measured & np.isin(symbols_upper, list(EWS_FLI1_TARGETS)))
                ews_fli1_controls = expression_matched_controls(pre_mean, measured, ews_fli1_index)
                baseline_top, predicted_top, ews_pre_parts, ews_pred_parts = [], [], [], []
                for start, stop, expression in aligned_batches(pre, mapping, transform, args.batch_size, args.max_pre_observations):
                    decoded = model.decode(torch.from_numpy(z_pred[start:stop]).to(device)).cpu().numpy()
                    baseline_top.append(expression[:, top_index].astype(np.float32))
                    predicted_top.append(decoded[:, top_index].astype(np.float32))
                    if len(ews_fli1_index) and len(ews_fli1_controls):
                        ews_pre_parts.append(
                            expression[:, ews_fli1_index].mean(axis=1) - expression[:, ews_fli1_controls].mean(axis=1)
                        )
                        ews_pred_parts.append(
                            decoded[:, ews_fli1_index].mean(axis=1) - decoded[:, ews_fli1_controls].mean(axis=1)
                        )
                baseline_top = np.concatenate(baseline_top)
                predicted_top = np.concatenate(predicted_top)
                spatial_delta = predicted_top - baseline_top
                for column, model_index in enumerate(top_index):
                    hotspot_rows.append({
                        "sample_id": sample.sample_id,
                        "gse_id": sample.gse_id,
                        "patient_key": sample.patient_key,
                        "gene_id": model_genes.iloc[model_index]["gene_id"],
                        "gene_symbol": model_genes.iloc[model_index]["gene_symbol"],
                        "mean_predicted_delta": float(spatial_delta[:, column].mean()),
                        "spatial_delta_morans_i": morans_i(spatial_delta[:, column], coords),
                    })
                spatial_var = model_genes.iloc[top_index][["gene_id", "gene_symbol"]].copy()
                spatial_var.index = spatial_var["gene_id"].astype(str)
                predicted_adata = ad.AnnData(X=predicted_top, obs=pre.obs.iloc[:n_pre].copy(), var=spatial_var)
                predicted_adata.layers["pretreatment"] = baseline_top
                predicted_adata.layers["predicted_delta"] = spatial_delta
                predicted_adata.obsm["spatial"] = coords
                if ews_pre_parts:
                    ews_pre = np.concatenate(ews_pre_parts).astype(np.float32)
                    ews_pred = np.concatenate(ews_pred_parts).astype(np.float32)
                    ews_delta = ews_pred - ews_pre
                    predicted_adata.obs["ewsfli1_module_pretreatment"] = ews_pre
                    predicted_adata.obs["ewsfli1_module_predicted_posttreatment"] = ews_pred
                    predicted_adata.obs["ewsfli1_module_predicted_delta"] = ews_delta
                    metric_rows[-1].update({
                        "ewsfli1_target_genes_measured": len(ews_fli1_index),
                        "ewsfli1_control_genes": len(ews_fli1_controls),
                        "ewsfli1_module_pretreatment_mean": float(ews_pre.mean()),
                        "ewsfli1_module_predicted_posttreatment_mean": float(ews_pred.mean()),
                        "ewsfli1_module_predicted_delta_mean": float(ews_delta.mean()),
                        "ewsfli1_module_predicted_delta_morans_i": morans_i(ews_delta, coords),
                        "ewsfli1_program_reversal_predicted": bool(ews_delta.mean() < 0),
                    })
                predicted_adata.uns["prediction_metadata"] = {
                    "sample_id": sample.sample_id,
                    "checkpoint": str(args.checkpoint),
                    "input": "pretreatment only",
                    "measured_posttreatment_used": False,
                    "expression_scale": transform,
                    "gene_selection": (
                        f"top {top_count} measured genes by absolute predicted pseudobulk change "
                        f"plus {len(mechanism_index)} measured mechanism-panel genes"
                    ),
                    "ewsfli1_score": (
                        "mean of measured 10-gene EWS::FLI1-induced target panel minus "
                        "baseline-expression-matched control-gene mean"
                    ),
                }
                predicted_h5ad = out / "predicted_h5ad" / f"{sample.sample_id}.h5ad"
                predicted_adata.write_h5ad(predicted_h5ad, compression="gzip")
                with gzip.open(out / "alignment" / f"{sample.sample_id}_pretreatment.tsv.gz", "wt", encoding="utf-8") as handle:
                    report.to_csv(handle, sep="\t", index=False)
                plot_shift(
                    out / "figures" / f"{sample.sample_id}_predicted_shift.png",
                    coords, shift, f"{sample.gse_id} {sample.patient_id}: predicted biological transition",
                )
                result_rows.append({
                    **sample._asdict(),
                    "gene_table": str(gene_path.relative_to(out)),
                    "latent_npz": f"latent/{sample.sample_id}.npz",
                    "predicted_h5ad": str(predicted_h5ad.relative_to(out)),
                })
            finally:
                pre.file.close()

    pd.DataFrame(metric_rows).to_csv(out / "sample_metrics.tsv", sep="\t", index=False)
    pd.DataFrame(hotspot_rows).to_csv(out / "spatial_gene_hotspots.tsv", sep="\t", index=False)
    pd.DataFrame(segment_rows).to_csv(out / "segment_prediction_manifest.tsv", sep="\t", index=False)
    pd.DataFrame(result_rows).to_csv(out / "prediction_manifest.tsv", sep="\t", index=False)
    model_genes.to_csv(out / "model_genes.tsv", sep="\t", index=False)
    write_json(out / "prediction_summary.json", {
        "checkpoint": str(args.checkpoint),
        "expression_input": "stored raw pretreatment counts aligned to checkpoint order, then checkpoint-declared transform",
        "expression_transform": transform,
        "prediction_output_scale": transform,
        "fine_tuning": False,
        "measured_posttreatment_loaded": False,
        "analysis_contrast": "pretreatment versus model-predicted posttreatment",
        "spatial_interpretation": "predicted state and predicted delta remain on pretreatment coordinates",
        "baseline_samples": len(metric_rows),
    })
    print(f"completed_baselines={len(metric_rows)} output={out}")


if __name__ == "__main__":
    main()
