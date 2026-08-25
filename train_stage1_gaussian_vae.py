#!/usr/bin/env python3
"""Train paper-faithful Stage 1 on raw-count H5AD pairs.

Raw counts remain stored on disk. The loader applies normalize_total(1e4)
followed by log1p, and this script trains a Gaussian VAE with a 128-dimensional
latent. Source and target cells are both reconstructed; MOSCOT mass is used
once as the marginal reconstruction weight.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from five_level_cell_world_model import GaussianExpressionVAE
from sample_paired_h5ad_dataloader import build_paired_h5ad_loader


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--checkpoint-dir", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=4096)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-2)
    p.add_argument("--kl-weight", type=float, default=1e-3)
    p.add_argument("--latent-dim", type=int, default=128)
    p.add_argument("--hidden-dim", type=int, default=512)
    p.add_argument("--depth", type=int, default=3)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--num-folds", type=int, default=5)
    p.add_argument("--fold-index", type=int, default=0)
    p.add_argument("--group-column", default="patient_id")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--max-train-steps", type=int, default=-1)
    p.add_argument("--max-val-steps", type=int, default=-1)
    p.add_argument("--device", default="auto")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def choose_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def weighted_mean(values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    return (values * weights).sum() / weights.sum().clamp_min(1e-8)


def vae_loss(
    model: GaussianExpressionVAE,
    batch: dict,
    device: torch.device,
    kl_weight: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x = batch["x"].to(device, non_blocking=True)
    y = batch["y"].to(device, non_blocking=True)
    weight = batch["weight"].to(device, non_blocking=True)
    expression = torch.cat([x, y], dim=0)
    weights = torch.cat([weight, weight], dim=0)
    out = model(expression)
    reconstruction = (out["reconstruction"] - expression).pow(2).mean(dim=1)
    kl = -0.5 * (1.0 + out["logvar"] - out["mu"].pow(2) - out["logvar"].exp()).mean(dim=1)
    reconstruction_loss = weighted_mean(reconstruction, weights)
    kl_loss = weighted_mean(kl, weights)
    return reconstruction_loss + kl_weight * kl_loss, reconstruction_loss, kl_loss


def run_epoch(
    model: GaussianExpressionVAE,
    loader,
    device: torch.device,
    kl_weight: float,
    optimizer: torch.optim.Optimizer | None,
    max_steps: int,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    totals = np.zeros(3, dtype=np.float64)
    steps = 0
    context = torch.enable_grad if training else torch.no_grad
    with context():
        for batch in loader:
            if max_steps >= 0 and steps >= max_steps:
                break
            if training:
                optimizer.zero_grad(set_to_none=True)
            loss, reconstruction, kl = vae_loss(model, batch, device, kl_weight)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite Stage-1 VAE loss")
            if training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            totals += [float(loss.detach().cpu()), float(reconstruction.detach().cpu()), float(kl.detach().cpu())]
            steps += 1
    if steps == 0:
        raise ValueError("Stage-1 loader produced no batches")
    mean = totals / steps
    return {"steps": steps, "loss": mean[0], "reconstruction": mean[1], "kl": mean[2]}


def save_checkpoint(path: Path, model: GaussianExpressionVAE, args, dataset, epoch: int, metrics: dict) -> None:
    torch.save(
        {
            "format": "stage1_gaussian_vae_log1p10k_v1",
            "vae_state_dict": model.state_dict(),
            "epoch": epoch,
            "metrics": metrics,
            "args": vars(args),
            "n_genes": dataset.n_genes,
            "gene_ids": dataset.gene_ids,
            "gene_symbols": dataset.gene_symbols,
            "expression_transform": "log1p_10k",
            "representation_type": "gaussian_vae",
        },
        path,
    )


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = choose_device(args.device)
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_path = args.checkpoint_dir / f"best_stage1_vae_fold{args.fold_index}.pt"
    history_path = args.checkpoint_dir / f"stage1_vae_fold{args.fold_index}_history.csv"
    config_path = args.checkpoint_dir / f"stage1_vae_fold{args.fold_index}_config.json"
    if not args.overwrite and (best_path.exists() or history_path.exists()):
        raise FileExistsError("Stage-1 outputs already exist; use --overwrite")

    loader_kwargs = dict(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        sample_by_probability=False,
        use_probability_as_weight=True,
        expression_transform="log1p_10k",
        num_folds=args.num_folds,
        fold_index=args.fold_index,
        group_column=args.group_column,
        num_workers=args.num_workers,
        seed=args.seed,
    )
    train_dataset, train_loader = build_paired_h5ad_loader(
        **loader_kwargs, split="train", shuffle_shards=True, shuffle_rows=True
    )
    val_dataset, val_loader = build_paired_h5ad_loader(
        **loader_kwargs, split="val", shuffle_shards=False, shuffle_rows=False
    )
    if train_dataset.gene_ids != val_dataset.gene_ids:
        raise ValueError("Stage-1 train/validation gene order differs")

    model = GaussianExpressionVAE(
        n_genes=train_dataset.n_genes,
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        depth=args.depth,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best = float("inf")
    rows = []
    for epoch in range(args.epochs):
        train = run_epoch(model, train_loader, device, args.kl_weight, optimizer, args.max_train_steps)
        val = run_epoch(model, val_loader, device, args.kl_weight, None, args.max_val_steps)
        row = {
            "epoch": epoch,
            **{f"train_{k}": v for k, v in train.items()},
            **{f"val_{k}": v for k, v in val.items()},
        }
        rows.append(row)
        print(
            f"epoch={epoch} train={train['loss']:.6g} val={val['loss']:.6g} "
            f"val_recon={val['reconstruction']:.6g} val_kl={val['kl']:.6g}"
        )
        if val["loss"] < best:
            best = val["loss"]
            save_checkpoint(best_path, model, args, train_dataset, epoch, row)

    with history_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    config_path.write_text(json.dumps(vars(args), indent=2, default=str) + "\n", encoding="utf-8")
    print(f"best_stage1_checkpoint={best_path} best_val_loss={best:.6g}")


if __name__ == "__main__":
    main()
