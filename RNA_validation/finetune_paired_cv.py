#!/usr/bin/env python3
"""Patient-grouped fine-tuning and evaluation on paired bulk RNA-seq."""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.stats import wilcoxon
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import average_precision_score, balanced_accuracy_score, matthews_corrcoef, roc_auc_score
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from RNA_validation.common import (
    build_world_model_from_checkpoint,
    choose_device,
    endpoint_metrics,
    load_checkpoint,
    require_empty_output,
    set_seed,
    write_json,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--prepared-dir", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--group-column", default="biological_patient_id")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--weight-decay", type=float, default=1e-3)
    p.add_argument("--flow-steps", type=int, default=8)
    p.add_argument("--endpoint-aux-weight", type=float, default=0.1)
    p.add_argument("--trainable", choices=["transition", "autoencoder", "all"], default="transition")
    p.add_argument("--ridge-components", type=int, default=32)
    p.add_argument("--save-fold-models", action="store_true")
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def configure_trainable(model: torch.nn.Module, choice: str) -> list[torch.nn.Parameter]:
    for parameter in model.parameters():
        parameter.requires_grad = choice == "all"
    if choice == "transition":
        for parameter in model.transition.parameters():
            parameter.requires_grad = True
    elif choice == "autoencoder":
        for parameter in model.autoencoder.parameters():
            parameter.requires_grad = True
    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        raise ValueError("No trainable parameters")
    return params


def predict(model, x: torch.Tensor, flow_steps: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    z_source = model.encode(x)
    if int(model.config.level) == 6:
        delta = torch.ones(len(x), device=x.device, dtype=x.dtype)
        z_pred = model.integrate_cellflow(z_source, delta, n_steps=flow_steps, method="euler")
    else:
        z_pred = model.transition(z_source, torch.ones(len(x), device=x.device, dtype=x.dtype))["z_pred"]
    return model.decode(z_pred), z_source, z_pred


def train_fold(model, x: np.ndarray, y: np.ndarray, indices: np.ndarray, args, device) -> list[dict]:
    params = configure_trainable(model, args.trainable)
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    rng = np.random.default_rng(args.seed)
    history = []
    model.train()
    if args.trainable == "transition":
        model.autoencoder.eval()
    elif args.trainable == "autoencoder":
        model.transition.eval()
    for epoch in range(args.epochs):
        order = rng.permutation(indices)
        losses = []
        for start in range(0, len(order), args.batch_size):
            idx = order[start:start + args.batch_size]
            xb = torch.from_numpy(x[idx]).to(device)
            yb = torch.from_numpy(y[idx]).to(device)
            optimizer.zero_grad(set_to_none=True)
            if int(model.config.level) == 6:
                z_source = model.encode(xb)
                z_target = model.encode(yb).detach()
                t = torch.rand(len(xb), device=device, dtype=z_source.dtype)
                z_t = (1 - t[:, None]) * z_source + t[:, None] * z_target
                delta = torch.ones(len(xb), device=device, dtype=z_source.dtype)
                velocity = model.predict_velocity(z_t, t, delta)
                loss_flow = F.mse_loss(velocity, z_target - z_source)
                if args.endpoint_aux_weight > 0:
                    z_end = model.integrate_cellflow(z_source, delta, n_steps=args.flow_steps, method="euler")
                    loss_endpoint = F.mse_loss(z_end, z_target)
                else:
                    loss_endpoint = torch.zeros((), device=device)
                loss = loss_flow + args.endpoint_aux_weight * loss_endpoint
            else:
                y_pred, _, _ = predict(model, xb, args.flow_steps)
                loss_flow = torch.zeros((), device=device)
                loss_endpoint = F.mse_loss(y_pred, yb)
                loss = loss_endpoint
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
            losses.append((float(loss.detach().cpu()), float(loss_flow.detach().cpu()), float(loss_endpoint.detach().cpu())))
        mean = np.mean(losses, axis=0)
        history.append({"epoch": epoch, "loss": mean[0], "flow_loss": mean[1], "endpoint_loss": mean[2]})
        print(f"epoch={epoch} loss={mean[0]:.6g}")
    return history


def ridge_prediction(x_train, y_train, x_test, n_components: int) -> np.ndarray:
    n = min(n_components, len(x_train) - 1, x_train.shape[1])
    if n < 1:
        return np.repeat(y_train.mean(axis=0, keepdims=True), len(x_test), axis=0)
    pca = PCA(n_components=n, svd_solver="randomized", random_state=0)
    z_train = pca.fit_transform(x_train)
    z_test = pca.transform(x_test)
    return Ridge(alpha=10.0).fit(z_train, y_train).predict(z_test).astype(np.float32)


def response_fit_predict(features: np.ndarray, labels: np.ndarray, train_idx: np.ndarray, test_idx: np.ndarray) -> np.ndarray:
    train = train_idx[labels[train_idx] >= 0]
    test = test_idx[labels[test_idx] >= 0]
    result = np.full(len(test_idx), np.nan, dtype=float)
    if len(train) < 4 or len(np.unique(labels[train])) < 2 or len(test) == 0:
        return result
    classifier = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, class_weight="balanced", C=0.1, solver="liblinear"),
    )
    classifier.fit(features[train], labels[train])
    positions = {global_i: local_i for local_i, global_i in enumerate(test_idx)}
    probabilities = classifier.predict_proba(features[test])[:, 1]
    for global_i, probability in zip(test, probabilities):
        result[positions[global_i]] = probability
    return result


def pca_response_fit_predict(x: np.ndarray, labels: np.ndarray, train_idx: np.ndarray, test_idx: np.ndarray, components: int) -> np.ndarray:
    train = train_idx[labels[train_idx] >= 0]
    test = test_idx[labels[test_idx] >= 0]
    result = np.full(len(test_idx), np.nan, dtype=float)
    if len(train) < 4 or len(np.unique(labels[train])) < 2 or len(test) == 0:
        return result
    n = min(components, len(train) - 1, x.shape[1])
    pca = PCA(n_components=n, svd_solver="randomized", random_state=0)
    train_x = pca.fit_transform(x[train])
    test_x = pca.transform(x[test])
    classifier = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, class_weight="balanced", C=0.1, solver="liblinear"),
    )
    classifier.fit(train_x, labels[train])
    positions = {global_i: local_i for local_i, global_i in enumerate(test_idx)}
    for global_i, probability in zip(test, classifier.predict_proba(test_x)[:, 1]):
        result[positions[global_i]] = probability
    return result


def summarize_finetuning(metrics: pd.DataFrame, metadata: pd.DataFrame, seed: int):
    """Paired held-out comparison; positive improvements always favor fine-tuning."""
    selected = metrics.loc[metrics["method"].isin(["world_pretrained", "world_finetuned"])].copy()
    value_columns = ["mse", "mae", "post_pearson", "delta_pearson"]
    wide = selected.pivot(index="row_index", columns="method", values=value_columns)
    wide.columns = [f"{metric}__{method}" for metric, method in wide.columns]
    rows = metadata.copy()
    rows["row_index"] = pd.to_numeric(rows["row_index"], errors="raise").astype(int)
    rows = rows.set_index("row_index")
    comparison = rows[["patient_id", "biological_patient_id", "source_dataset_id"]].join(wide, how="inner")
    for metric in value_columns:
        pretrained = pd.to_numeric(comparison[f"{metric}__world_pretrained"], errors="coerce")
        finetuned = pd.to_numeric(comparison[f"{metric}__world_finetuned"], errors="coerce")
        if metric in {"mse", "mae"}:
            comparison[f"{metric}_improvement"] = pretrained - finetuned
        else:
            comparison[f"{metric}_improvement"] = finetuned - pretrained

    improvement_columns = [f"{metric}_improvement" for metric in value_columns]
    patient = comparison.groupby(
        ["biological_patient_id", "patient_id", "source_dataset_id"], as_index=False
    )[improvement_columns].mean()
    rng = np.random.default_rng(seed)
    summary_rows = []
    for column in improvement_columns:
        values = pd.to_numeric(patient[column], errors="coerce").dropna().to_numpy(float)
        if len(values):
            boot = np.asarray([
                np.mean(values[rng.integers(0, len(values), len(values))]) for _ in range(2000)
            ])
            ci_low, ci_high = np.percentile(boot, [2.5, 97.5])
        else:
            ci_low = ci_high = np.nan
        nonzero = values[values != 0]
        p_value = float(wilcoxon(nonzero).pvalue) if len(nonzero) >= 2 else np.nan
        summary_rows.append({
            "metric": column.removesuffix("_improvement"),
            "positive_favors": "world_finetuned",
            "n_patients": len(values),
            "mean_improvement": float(np.mean(values)) if len(values) else np.nan,
            "median_improvement": float(np.median(values)) if len(values) else np.nan,
            "bootstrap_ci_low": float(ci_low),
            "bootstrap_ci_high": float(ci_high),
            "finetuned_win_rate": float(np.mean(values > 0)) if len(values) else np.nan,
            "tie_rate": float(np.mean(values == 0)) if len(values) else np.nan,
            "wilcoxon_p_value": p_value,
        })
    return comparison.reset_index(), pd.DataFrame(summary_rows)


def main() -> None:
    args = parse_args()
    out = require_empty_output(args.output_dir, args.overwrite)
    set_seed(args.seed)
    device = choose_device(args.device)
    arrays = np.load(args.prepared_dir / "paired_expression.npz")
    x = arrays["x_pre"].astype(np.float32)
    y = arrays["y_post"].astype(np.float32)
    metadata = pd.read_csv(args.prepared_dir / "patients.tsv", sep="\t", dtype=str, keep_default_na=False)
    if len(metadata) != len(x) or x.shape != y.shape:
        raise ValueError("Prepared expression and metadata dimensions disagree")
    if args.group_column not in metadata.columns:
        raise ValueError(f"Missing group column: {args.group_column}")
    groups = metadata[args.group_column].to_numpy()
    labels = pd.to_numeric(metadata["response_binary"], errors="coerce").fillna(-1).to_numpy(dtype=int)
    stratify = np.where(labels >= 0, labels, 2)
    try:
        splitter = StratifiedGroupKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
        splits = list(splitter.split(x, stratify, groups))
    except ValueError:
        splitter = GroupKFold(n_splits=args.folds)
        splits = list(splitter.split(x, stratify, groups))

    checkpoint = load_checkpoint(args.checkpoint)
    methods = ["world_pretrained", "world_finetuned", "identity", "mean_delta", "pca_ridge"]
    predictions = {name: np.zeros_like(y, dtype=np.float32) for name in methods}
    z_source_all = np.zeros((len(x), int(checkpoint.get("args", {}).get("latent_dim", 128))), dtype=np.float32)
    z_residual_all = np.zeros_like(z_source_all)
    response_probabilities = {
        "world_finetuned": np.full(len(x), np.nan, dtype=float),
        "world_pretrained": np.full(len(x), np.nan, dtype=float),
        "world_pretrained_source_only": np.full(len(x), np.nan, dtype=float),
        "pca_expression": np.full(len(x), np.nan, dtype=float),
    }
    fold_assignment = np.full(len(x), -1, dtype=int)
    histories = []

    for fold, (train_idx, test_idx) in enumerate(splits):
        if set(groups[train_idx]) & set(groups[test_idx]):
            raise RuntimeError("Patient leakage detected in generated folds")
        print(f"fold={fold} train={len(train_idx)} test={len(test_idx)}")
        fold_assignment[test_idx] = fold
        base_model = build_world_model_from_checkpoint(checkpoint, device).eval()
        with torch.no_grad():
            xb = torch.from_numpy(x[test_idx]).to(device)
            base_pred, base_test_source, base_test_pred = predict(base_model, xb, args.flow_steps)
            predictions["world_pretrained"][test_idx] = base_pred.cpu().numpy()
            train_xb = torch.from_numpy(x[train_idx]).to(device)
            _, base_train_source, base_train_pred = predict(base_model, train_xb, args.flow_steps)
            base_features = np.zeros((len(x), z_source_all.shape[1] * 2), dtype=np.float32)
            base_features[train_idx] = np.concatenate(
                [base_train_source.cpu().numpy(), (base_train_pred - base_train_source).cpu().numpy()], axis=1
            )
            base_features[test_idx] = np.concatenate(
                [base_test_source.cpu().numpy(), (base_test_pred - base_test_source).cpu().numpy()], axis=1
            )
            base_source_features = np.zeros((len(x), z_source_all.shape[1]), dtype=np.float32)
            base_source_features[train_idx] = base_train_source.cpu().numpy()
            base_source_features[test_idx] = base_test_source.cpu().numpy()
        model = build_world_model_from_checkpoint(checkpoint, device)
        history = train_fold(model, x, y, train_idx, args, device)
        for row in history:
            histories.append({"fold": fold, **row})
        model.eval()
        with torch.no_grad():
            xb = torch.from_numpy(x[test_idx]).to(device)
            pred, z_source, z_pred = predict(model, xb, args.flow_steps)
            predictions["world_finetuned"][test_idx] = pred.cpu().numpy()
            z_source_all[test_idx] = z_source.cpu().numpy()
            z_residual_all[test_idx] = (z_pred - z_source).cpu().numpy()
            train_xb = torch.from_numpy(x[train_idx]).to(device)
            _, train_z_source, train_z_pred = predict(model, train_xb, args.flow_steps)
            local_features = np.zeros((len(x), z_source_all.shape[1] * 2), dtype=np.float32)
            local_features[train_idx] = np.concatenate(
                [train_z_source.cpu().numpy(), (train_z_pred - train_z_source).cpu().numpy()], axis=1
            )
            local_features[test_idx] = np.concatenate(
                [z_source.cpu().numpy(), (z_pred - z_source).cpu().numpy()], axis=1
            )
        predictions["identity"][test_idx] = x[test_idx]
        predictions["mean_delta"][test_idx] = x[test_idx] + np.mean(y[train_idx] - x[train_idx], axis=0, keepdims=True)
        predictions["pca_ridge"][test_idx] = ridge_prediction(x[train_idx], y[train_idx], x[test_idx], args.ridge_components)
        response_probabilities["world_finetuned"][test_idx] = response_fit_predict(local_features, labels, train_idx, test_idx)
        response_probabilities["world_pretrained"][test_idx] = response_fit_predict(base_features, labels, train_idx, test_idx)
        response_probabilities["world_pretrained_source_only"][test_idx] = response_fit_predict(
            base_source_features, labels, train_idx, test_idx
        )
        response_probabilities["pca_expression"][test_idx] = pca_response_fit_predict(
            x, labels, train_idx, test_idx, args.ridge_components
        )
        if args.save_fold_models:
            torch.save({"model_state_dict": model.state_dict(), "fold": fold, "args": vars(args)}, out / f"fold_{fold}_model.pt")
        del model, base_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    metadata["fold"] = fold_assignment
    for name, values in response_probabilities.items():
        metadata[f"{name}_response_probability"] = values
    metadata.to_csv(out / "patient_predictions.tsv", sep="\t", index=False)
    pd.DataFrame(histories).to_csv(out / "training_history.tsv", sep="\t", index=False)
    metric_tables = []
    for method, predicted in predictions.items():
        table = endpoint_metrics(y, predicted, x)
        table.insert(0, "method", method)
        table["patient_id"] = metadata["patient_id"]
        table["fold"] = fold_assignment
        metric_tables.append(table)
    metrics = pd.concat(metric_tables, ignore_index=True)
    metrics.to_csv(out / "expression_metrics_by_patient.tsv", sep="\t", index=False)
    summary = metrics.groupby("method")[["mse", "mae", "post_pearson", "delta_pearson"]].agg(["mean", "median", "count"])
    summary.columns = ["_".join(c) for c in summary.columns]
    summary.reset_index().to_csv(out / "expression_metrics_summary.tsv", sep="\t", index=False)
    fine_by_record, fine_summary = summarize_finetuning(metrics, metadata, args.seed)
    fine_by_record.to_csv(out / "finetuning_comparison_by_record.tsv", sep="\t", index=False)
    fine_summary.to_csv(out / "finetuning_comparison_summary.tsv", sep="\t", index=False)

    response_summary = {}
    for name, probability in response_probabilities.items():
        valid = (labels >= 0) & np.isfinite(probability)
        if valid.sum() and len(np.unique(labels[valid])) == 2:
            hard = (probability[valid] >= 0.5).astype(int)
            response_summary[name] = {
                "n": int(valid.sum()),
                "auroc": float(roc_auc_score(labels[valid], probability[valid])),
                "auprc": float(average_precision_score(labels[valid], probability[valid])),
                "balanced_accuracy": float(balanced_accuracy_score(labels[valid], hard)),
                "mcc": float(matthews_corrcoef(labels[valid], hard)),
            }
    np.savez_compressed(
        out / "cross_validated_predictions.npz",
        x_pre=x,
        y_post=y,
        z_source=z_source_all,
        z_residual=z_residual_all,
        **{f"pred_{key}": value for key, value in predictions.items()},
    )
    write_json(out / "run_summary.json", {
        "checkpoint": str(args.checkpoint),
        "prepared_dir": str(args.prepared_dir),
        "split_unit": args.group_column,
        "split_policy": "patient-grouped; datasets may occur in both train and validation by explicit user request",
        "folds": args.folds,
        "n_records": len(x),
        "n_unique_groups": int(len(np.unique(groups))),
        "trainable": args.trainable,
        "expression_contract": "raw counts are stored; validation preparation applies the checkpoint-declared transform",
        "response": response_summary,
        "comparators": methods,
        "finetuning_comparison": "finetuning_comparison_summary.tsv; positive improvement favors world_finetuned",
    })


if __name__ == "__main__":
    main()
