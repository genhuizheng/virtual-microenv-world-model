#!/usr/bin/env python3
"""Run the fixed CellFlow checkpoint over prepared TCGA tumors."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from RNA_validation.common import build_world_model_from_checkpoint, choose_device, load_checkpoint, require_empty_output, write_json
from RNA_validation.finetune_paired_cv import predict


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--prepared-dir", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--flow-steps", type=int, default=8)
    p.add_argument("--device", default="auto")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out = require_empty_output(args.output_dir, args.overwrite)
    (out / "shards").mkdir(exist_ok=True)
    checkpoint = load_checkpoint(args.checkpoint)
    device = choose_device(args.device)
    model = build_world_model_from_checkpoint(checkpoint, device).eval()
    manifest = pd.read_csv(args.prepared_dir / "manifest.tsv", sep="\t")
    result_manifest = []
    feature_tables = []
    with torch.no_grad():
        for row in manifest.itertuples(index=False):
            prepared_npz = np.load(args.prepared_dir / row.expression_npz)
            x = prepared_npz["x_pre"].astype(np.float32)
            pred_parts, source_parts, residual_parts = [], [], []
            for start in range(0, len(x), args.batch_size):
                xb = torch.from_numpy(x[start:start + args.batch_size]).to(device)
                pred, z_source, z_pred = predict(model, xb, args.flow_steps)
                pred_parts.append(pred.cpu().numpy())
                source_parts.append(z_source.cpu().numpy())
                residual_parts.append((z_pred - z_source).cpu().numpy())
            pred = np.concatenate(pred_parts).astype(np.float32)
            z_source = np.concatenate(source_parts).astype(np.float32)
            z_residual = np.concatenate(residual_parts).astype(np.float32)
            result_name = f"{row.dataset_id}_predictions.npz"
            np.savez_compressed(out / "shards" / result_name, x_pre=x, pred_post=pred, z_source=z_source, z_residual=z_residual)
            source_meta = args.prepared_dir / row.metadata_tsv
            meta_name = f"{row.dataset_id}_patients.tsv"
            shutil.copyfile(source_meta, out / "shards" / meta_name)
            meta = pd.read_csv(source_meta, sep="\t", dtype=str, keep_default_na=False)
            meta["predicted_delta_l2"] = np.linalg.norm(pred - x, axis=1)
            meta["latent_residual_l2"] = np.linalg.norm(z_residual, axis=1)
            meta["source_latent_l2"] = np.linalg.norm(z_source, axis=1)
            feature_tables.append(meta[["patient_id", "dataset_id", "predicted_delta_l2", "latent_residual_l2", "source_latent_l2"]])
            result_manifest.append({
                "dataset_id": row.dataset_id,
                "prediction_npz": f"shards/{result_name}",
                "metadata_tsv": f"shards/{meta_name}",
                "n_patients": len(x),
            })
            print(f"inferred {row.dataset_id}: {len(x)} patients")
    pd.DataFrame(result_manifest).to_csv(out / "manifest.tsv", sep="\t", index=False)
    pd.concat(feature_tables, ignore_index=True).to_csv(out / "patient_prediction_features.tsv", sep="\t", index=False)
    shutil.copyfile(args.prepared_dir / "model_genes.tsv", out / "model_genes.tsv")
    write_json(out / "run_summary.json", {
        "prepared_dir": str(args.prepared_dir),
        "checkpoint": str(args.checkpoint),
        "n_projects": len(result_manifest),
        "n_patients": int(sum(x["n_patients"] for x in result_manifest)),
        "expression_contract": "raw counts end to end; no CPM, log1p, clipping, or composition rescaling",
        "interpretation": "Exploratory generic transition from an unconditioned treatment model; not a treatment-specific causal prediction.",
    })


if __name__ == "__main__":
    main()
