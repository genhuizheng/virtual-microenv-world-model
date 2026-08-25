#!/usr/bin/env python3
"""Use official scGPT embeddings as a pretreatment response comparator."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, balanced_accuracy_score, matthews_corrcoef, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from RNA_validation.common import require_empty_output, write_json
from RNA_validation.torchtext_compat import install_torchtext_compat


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--prepared-dir", type=Path, required=True)
    p.add_argument(
        "--raw-input-dir", type=Path, required=True,
        help="Original standardized_rnaseq directory; scGPT is mapped independently from raw pretreatment counts",
    )
    p.add_argument("--scgpt-model-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--folds-file", type=Path, default=None, help="patient_predictions.tsv from finetune_paired_cv.py")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--max-length", type=int, default=1200)
    p.add_argument("--device", default="cuda")
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def load_vocab_tokens(vocab_path: Path) -> tuple[dict[str, str], dict[str, int]]:
    payload = json.loads(vocab_path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        token_to_id = {str(token): int(idx) for token, idx in payload.items()}
    elif isinstance(payload, list):
        token_to_id = {str(token): idx for idx, token in enumerate(payload)}
    else:
        raise ValueError(f"Unexpected scGPT vocabulary format: {vocab_path}")
    upper_to_token: dict[str, str] = {}
    for token in token_to_id:
        upper = token.strip().upper()
        if upper and upper not in upper_to_token:
            upper_to_token[upper] = token
    return upper_to_token, token_to_id


def load_raw_pretreatment(
    input_dir: Path, metadata: pd.DataFrame, vocab_path: Path
) -> tuple[np.ndarray, list[str], pd.DataFrame]:
    """Build patients x scGPT-vocabulary genes without using world-model gene order."""
    upper_to_token, token_to_id = load_vocab_tokens(vocab_path)
    chunks: list[pd.DataFrame] = []
    audits: list[dict[str, object]] = []
    for dataset_id, rows in metadata.groupby("dataset_id", sort=True):
        matrix_path = input_dir / "counts" / f"{dataset_id}_counts.tsv"
        requested = rows["pre_sample_id"].astype(str).tolist()
        wanted = set(requested)
        usecols = lambda c: c in {"gene_id", "gene_symbol"} or c in wanted
        matrix = pd.read_csv(
            matrix_path, sep="\t", usecols=usecols,
            dtype={"gene_id": str, "gene_symbol": str},
        )
        missing_samples = wanted - set(matrix.columns)
        if missing_samples:
            raise ValueError(f"Missing pretreatment samples in {matrix_path}: {sorted(missing_samples)[:5]}")
        source_symbols = matrix["gene_symbol"].astype(str).str.strip().str.upper()
        canonical = source_symbols.map(upper_to_token)
        matched = canonical.notna()
        if not matched.any():
            raise ValueError(f"No genes from {matrix_path} matched the scGPT vocabulary")
        gene_by_sample = pd.DataFrame(
            matrix.loc[matched, requested].to_numpy(dtype=np.float32),
            index=canonical[matched].tolist(),
            columns=requested,
        )
        # Counts from duplicate source rows mapping to one symbol must be summed.
        gene_by_sample = gene_by_sample.groupby(level=0, sort=False).sum()
        sample_by_gene = gene_by_sample.T
        sample_by_gene.index = rows.index
        chunks.append(sample_by_gene)
        audits.append({
            "dataset_id": dataset_id,
            "n_patients": len(rows),
            "source_gene_rows": len(matrix),
            "matched_source_rows": int(matched.sum()),
            "unique_scgpt_genes": int(gene_by_sample.shape[0]),
        })
    frame = pd.concat(chunks, axis=0, join="outer").fillna(0.0).reindex(metadata.index)
    symbols = sorted(frame.columns, key=lambda token: token_to_id[token])
    frame = frame.loc[:, symbols]
    return frame.to_numpy(dtype=np.float32), symbols, pd.DataFrame(audits)


def main() -> None:
    args = parse_args()
    out = require_empty_output(args.output_dir, args.overwrite)
    install_torchtext_compat()
    from scgpt.tasks import embed_data

    metadata = pd.read_csv(args.prepared_dir / "patients.tsv", sep="\t", dtype=str, keep_default_na=False)
    x, symbols, gene_audit = load_raw_pretreatment(
        args.raw_input_dir, metadata, args.scgpt_model_dir / "vocab.json"
    )
    gene_audit.to_csv(out / "scgpt_gene_mapping_by_dataset.tsv", sep="\t", index=False)
    obs = metadata[["patient_id", "biological_patient_id", "response_binary"]].copy()
    obs.index = [f"scgpt_row_{i}" for i in range(len(obs))]
    var = pd.DataFrame({"feature_name": symbols}, index=symbols)
    adata = ad.AnnData(X=sparse.csr_matrix(x), obs=obs, var=var)
    embedded = embed_data(
        adata,
        args.scgpt_model_dir,
        gene_col="feature_name",
        max_length=args.max_length,
        batch_size=args.batch_size,
        obs_to_save=["patient_id", "biological_patient_id", "response_binary"],
        device=args.device,
        use_fast_transformer=False,
        return_new_adata=True,
    )
    embeddings = embedded.X.toarray() if sparse.issparse(embedded.X) else np.asarray(embedded.X)
    np.save(out / "scgpt_embeddings.npy", embeddings.astype(np.float32))
    labels = pd.to_numeric(metadata["response_binary"], errors="coerce").fillna(-1).to_numpy(dtype=int)
    groups = metadata["biological_patient_id"].to_numpy()
    fold = np.full(len(metadata), -1, dtype=int)
    if args.folds_file:
        assigned = pd.read_csv(args.folds_file, sep="\t", dtype=str, keep_default_na=False)
        mapping = dict(zip(assigned["patient_id"], pd.to_numeric(assigned["fold"], errors="raise").astype(int)))
        fold = metadata["patient_id"].map(mapping).to_numpy(dtype=int)
    else:
        stratify = np.where(labels >= 0, labels, 2)
        splitter = StratifiedGroupKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
        for fold_i, (_, test_idx) in enumerate(splitter.split(embeddings, stratify, groups)):
            fold[test_idx] = fold_i
    probability = np.full(len(metadata), np.nan, dtype=float)
    for fold_i in sorted(set(fold)):
        test_idx = np.where(fold == fold_i)[0]
        train_idx = np.where(fold != fold_i)[0]
        train = train_idx[labels[train_idx] >= 0]
        test = test_idx[labels[test_idx] >= 0]
        if len(test) == 0 or len(np.unique(labels[train])) < 2:
            continue
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, C=0.1, class_weight="balanced", solver="liblinear"),
        )
        model.fit(embeddings[train], labels[train])
        probability[test] = model.predict_proba(embeddings[test])[:, 1]
    result = metadata.copy()
    result["fold"] = fold
    result["scgpt_response_probability"] = probability
    result.to_csv(out / "patient_predictions.tsv", sep="\t", index=False)
    valid = (labels >= 0) & np.isfinite(probability)
    metrics = {}
    if valid.sum() and len(np.unique(labels[valid])) == 2:
        hard = (probability[valid] >= 0.5).astype(int)
        metrics = {
            "n": int(valid.sum()),
            "auroc": float(roc_auc_score(labels[valid], probability[valid])),
            "auprc": float(average_precision_score(labels[valid], probability[valid])),
            "balanced_accuracy": float(balanced_accuracy_score(labels[valid], hard)),
            "mcc": float(matthews_corrcoef(labels[valid], hard)),
        }
    write_json(out / "run_summary.json", {
        "scgpt_model_dir": str(args.scgpt_model_dir),
        "raw_input_dir": str(args.raw_input_dir),
        "n_scgpt_vocab_genes_observed": len(symbols),
        "gene_mapping": "source gene symbols mapped directly to scGPT vocab; independent of world-model gene order",
        "task": "pretreatment embedding plus patient-grouped logistic response classifier",
        "metrics": metrics,
        "warning": "scGPT is not used as a post-treatment expression generator; this is an embedding/classification comparison.",
    })


if __name__ == "__main__":
    main()
