#!/usr/bin/env python3
"""
Train and validate one of the one-step temporal cell world-model levels.

Current screening design:
    primary training loss: weighted pair MSE in latent space
    validation selection: val_loss_pair
    secondary metrics: MMD, Sinkhorn W2, paired drift, downhill regularizer
"""

from __future__ import annotations

from pathlib import Path
import argparse
import csv
import json
import math
import time

import torch

from sample_paired_h5ad_dataloader import build_paired_h5ad_loader
from five_level_cell_world_model import build_model, weighted_mse
from chreode_loss import ChreodeLossConfig, chreode_population_loss


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument(
        "--level",
        type=int,
        choices=[0, 1, 2, 3, 4, 5, 6],
        default=2,
        help="0 is the non-learned identity baseline (predicts no change); it has no "
        "trainable transition parameters and serves as the sanity floor for every "
        "other level.",
    )
    parser.add_argument(
        "--variant",
        choices=[
            "default",
            "tokenized_dit",
            "stochastic_dit",
            "cross_attention_dit",
            "hierarchical_dit",
            "consistency_dit",
            "rectified_dit",
            "diffusion_denoising_dit",
        ],
        default="default",
        help="Level-5 architecture variant. Use default for levels 1-4.",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument(
        "--latent-dim",
        type=int,
        default=None,
        help="Defaults to 128, except with --stage1-scvi-checkpoint, where it is "
        "adopted from the checkpoint: the transition operates inside that latent, so "
        "the dimension is part of the Stage-1 contract rather than a free choice. "
        "Passing a value that contradicts the checkpoint is an error.",
    )
    parser.add_argument("--expression-hidden-dim", type=int, default=1024)
    parser.add_argument("--expression-depth", type=int, default=3)
    parser.add_argument(
        "--stage1-vae-checkpoint",
        type=Path,
        default=None,
        help="Required for paper-faithful Stage 2; loads and freezes the Gaussian VAE.",
    )
    parser.add_argument(
        "--stage1-scvi-checkpoint",
        type=Path,
        default=None,
        help="Loads and freezes a Stage-1 scVI encoder/decoder (see train_stage1_scvi.py). "
        "Mutually exclusive with --stage1-vae-checkpoint; requires --expression-transform none.",
    )
    parser.add_argument(
        "--representation-type",
        choices=["legacy_mlp", "gaussian_vae", "scvi"],
        default="legacy_mlp",
    )
    parser.add_argument("--model-dim", type=int, default=256)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--num-tokens", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.1)

    parser.add_argument("--time-delta-col", type=str, default=None)
    parser.add_argument("--default-time-delta", type=float, default=1.0)
    parser.add_argument(
        "--expression-transform",
        type=str,
        default=None,
        help="Defaults to 'none', except with --stage1-scvi-checkpoint, where it is "
        "adopted from the checkpoint so the frozen encoder always receives the units it "
        "was fitted on. Passing a value that contradicts the checkpoint is an error.",
    )
    parser.add_argument("--group-column", type=str, default=None)
    parser.add_argument(
        "--technology-lookup",
        type=Path,
        default=None,
        help="TSV with dataset_id/technology columns. Required only when the Stage-1 "
        "scVI checkpoint was trained with a technology filter or --batch-key technology; "
        "the filter and batch key themselves are read from the checkpoint so the two "
        "stages cannot disagree.",
    )
    parser.add_argument("--sample-by-probability", action="store_true")
    parser.add_argument("--no-probability-weight", action="store_true")

    parser.add_argument("--num-folds", type=int, default=1)
    parser.add_argument("--fold-index", type=int, default=0)

    parser.add_argument("--loss-mode", choices=["pair_mse", "chreode", "expression_mse", "cellflow"], default="pair_mse")
    parser.add_argument("--loss-tag", type=str, default="manual", help="Human-readable tag for loss-parameter screening.")
    parser.add_argument("--lambda-mmd", type=float, default=1.0)
    parser.add_argument("--lambda-w2", type=float, default=1.0)
    parser.add_argument("--lambda-drift", type=float, default=1.0)
    parser.add_argument("--lambda-down", type=float, default=0.1)
    parser.add_argument("--sinkhorn-epsilon", type=float, default=0.1)
    parser.add_argument("--sinkhorn-iters", type=int, default=100)
    parser.add_argument("--expression-loss-weight", type=float, default=0.0)
    parser.add_argument("--latent-loss-weight", type=float, default=0.0)
    parser.add_argument("--source-recon-loss-weight", type=float, default=0.0)
    parser.add_argument("--cellflow-sigma", type=float, default=0.0)
    parser.add_argument("--cellflow-eval-time", type=float, default=0.5)
    parser.add_argument("--cellflow-integration-steps", type=int, default=8)
    parser.add_argument("--cellflow-integration-method", choices=["euler", "heun"], default="euler")
    parser.add_argument("--cellflow-pair-aux-weight", type=float, default=0.0)
    parser.add_argument("--cellflow-population-aux-weight", type=float, default=0.0)

    parser.add_argument("--checkpoint-dir", type=Path, default=None)
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--save-each-epoch", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--validate-every-epochs",
        type=int,
        default=1,
        help="Validate after epoch 1, every N completed epochs, and the final epoch.",
    )
    parser.add_argument(
        "--max-val-steps",
        type=int,
        default=-1,
        help="Maximum validation batches per validation epoch; -1 uses the full loader.",
    )
    parser.add_argument("--device", type=str, default="auto")
    return parser.parse_args()


def choose_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def append_csv(path: Path, fieldnames: list[str], row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def save_checkpoint(
    checkpoint_dir: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    args: argparse.Namespace,
    step: int,
    epoch: int,
    dataset,
    name: str | None = None,
) -> Path:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    variant_part = f"_{args.variant}" if args.variant != "default" else ""
    if name is None:
        name = f"world_model_level{args.level}{variant_part}_fold{args.fold_index}_step{step:07d}.pt"
    path = checkpoint_dir / name
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            # None for the level-0 identity baseline, which has no optimizer.
            "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
            "step": step,
            "epoch": epoch,
            "args": vars(args),
            "n_genes": dataset.n_genes,
            "gene_ids": dataset.gene_ids,
            "gene_symbols": dataset.gene_symbols,
        },
        path,
    )
    return path


def empty_stats() -> dict[str, float]:
    return {
        "n_steps": 0,
        "loss": 0.0,
        "loss_pair": 0.0,
        "loss_cellflow": 0.0,
        "loss_cellflow_pair_aux": 0.0,
        "loss_cellflow_population_aux": 0.0,
        "metric_mmd": 0.0,
        "metric_w2": 0.0,
        "metric_drift": 0.0,
        "metric_down": 0.0,
        "loss_expr": 0.0,
        "loss_latent_aux": 0.0,
        "loss_source_recon": 0.0,
    }


def add_stats(stats: dict[str, float], losses: dict[str, torch.Tensor]) -> None:
    stats["n_steps"] += 1
    for key in stats:
        if key == "n_steps":
            continue
        stats[key] += float(losses[key].detach().cpu())


def mean_stats(stats: dict[str, float]) -> dict[str, float]:
    n = max(int(stats["n_steps"]), 1)
    return {key: (value if key == "n_steps" else value / n) for key, value in stats.items()}


def make_loss_fields() -> list[str]:
    return [
        "split",
        "level",
        "variant",
        "loss_mode",
        "loss_tag",
        "fold_index",
        "num_folds",
        "epoch",
        "step",
        "n_steps",
        "loss",
        "loss_pair",
        "loss_cellflow",
        "loss_cellflow_pair_aux",
        "loss_cellflow_population_aux",
        "metric_mmd",
        "metric_w2",
        "metric_drift",
        "metric_down",
        "loss_expr",
        "loss_latent_aux",
        "loss_source_recon",
    ]


def compute_losses(
    model: torch.nn.Module,
    batch: dict,
    device: torch.device,
    args: argparse.Namespace,
    chreode_config: ChreodeLossConfig,
) -> dict[str, torch.Tensor]:
    x = batch["x"].to(device, non_blocking=True)
    y = batch["y"].to(device, non_blocking=True)
    delta = batch["delta"].to(device, non_blocking=True)
    weight = batch["weight"].to(device, non_blocking=True)
    # Source and target of a pair are the same patient in the same study, so a
    # single per-row batch label applies to both sides.
    batch_categories = batch.get("batch_category")

    freeze_autoencoder = bool(getattr(args, "stage1_vae_checkpoint", None)) or bool(
        getattr(args, "stage1_scvi_checkpoint", None)
    )
    if freeze_autoencoder:
        with torch.no_grad():
            z_target = model.encode(y, batch_categories)
    else:
        z_target = model.encode(y, batch_categories).detach()

    if args.loss_mode == "cellflow":
        if args.level != 6:
            raise ValueError("--loss-mode cellflow is only intended for --level 6.")
        if freeze_autoencoder:
            with torch.no_grad():
                z_source = model.encode(x, batch_categories)
        else:
            z_source = model.encode(x, batch_categories)
        z_pred = model.integrate_cellflow(
            z_source,
            delta,
            n_steps=args.cellflow_integration_steps,
            method=args.cellflow_integration_method,
        )
        out = {
            "y_pred": None,
            "z_source": z_source,
            "z_pred": z_pred,
            "delta": delta,
            "residual": z_pred - z_source,
        }
    else:
        out = model(x, delta, batch_categories)

    metric_parts = chreode_population_loss(out, z_target, weight, chreode_config)

    zero = torch.zeros((), device=device)
    loss_pair = weighted_mse(out["z_pred"], z_target, weight)
    loss_expr = zero
    loss_cellflow = zero
    loss_cellflow_pair_aux = zero
    loss_cellflow_population_aux = zero

    if args.loss_mode == "pair_mse":
        loss = loss_pair
    elif args.loss_mode == "chreode":
        loss = metric_parts["loss"]
    elif args.loss_mode == "cellflow":
        if model.training:
            flow_time = torch.rand(x.shape[0], device=device, dtype=out["z_source"].dtype)
        else:
            flow_time = torch.full(
                (x.shape[0],),
                float(args.cellflow_eval_time),
                device=device,
                dtype=out["z_source"].dtype,
            )
        z_source = out["z_source"]
        z_t = (1.0 - flow_time.view(-1, 1)) * z_source + flow_time.view(-1, 1) * z_target
        if model.training and args.cellflow_sigma > 0:
            z_t = z_t + float(args.cellflow_sigma) * torch.randn_like(z_t)
        target_velocity = z_target - z_source
        pred_velocity = model.predict_velocity(z_t, flow_time, delta)
        loss_cellflow = weighted_mse(pred_velocity, target_velocity, weight)
        loss = loss_cellflow
        if args.cellflow_pair_aux_weight > 0:
            loss_cellflow_pair_aux = loss_pair
            loss = loss + args.cellflow_pair_aux_weight * loss_cellflow_pair_aux
        if args.cellflow_population_aux_weight > 0:
            loss_cellflow_population_aux = (
                args.lambda_mmd * metric_parts["loss_mmd"]
                + args.lambda_w2 * metric_parts["loss_w2"]
                + args.lambda_drift * metric_parts["loss_drift"]
                + args.lambda_down * metric_parts["loss_down"]
            )
            loss = loss + args.cellflow_population_aux_weight * loss_cellflow_population_aux
    else:
        loss_expr = weighted_mse(out["y_pred"], y, weight)
        loss = loss_expr

    loss_latent_aux = zero
    if args.latent_loss_weight > 0:
        loss_latent_aux = weighted_mse(out["z_pred"], z_target, weight)
        loss = loss + args.latent_loss_weight * loss_latent_aux

    loss_source_recon = zero
    if args.source_recon_loss_weight > 0:
        if freeze_autoencoder:
            raise ValueError("Source reconstruction loss is incompatible with a frozen Stage-1 VAE")
        x_recon = model.decode(out["z_source"], batch_categories)
        loss_source_recon = weighted_mse(x_recon, x, weight)
        loss = loss + args.source_recon_loss_weight * loss_source_recon

    if args.expression_loss_weight > 0:
        if freeze_autoencoder:
            raise ValueError("Expression loss is incompatible with a frozen Stage-1 VAE")
        if out["y_pred"] is None:
            out["y_pred"] = model.decode(out["z_pred"], batch_categories)
        loss_expr = weighted_mse(out["y_pred"], y, weight)
        loss = loss + args.expression_loss_weight * loss_expr

    return {
        "loss": loss,
        "loss_pair": loss_pair,
        "loss_cellflow": loss_cellflow,
        "loss_cellflow_pair_aux": loss_cellflow_pair_aux,
        "loss_cellflow_population_aux": loss_cellflow_population_aux,
        "metric_mmd": metric_parts["loss_mmd"],
        "metric_w2": metric_parts["loss_w2"],
        "metric_drift": metric_parts["loss_drift"],
        "metric_down": metric_parts["loss_down"],
        "loss_expr": loss_expr,
        "loss_latent_aux": loss_latent_aux,
        "loss_source_recon": loss_source_recon,
    }


def log_row(args: argparse.Namespace, split: str, epoch: int, step: int, values: dict[str, float]) -> dict:
    return {
        "split": split,
        "level": args.level,
        "variant": args.variant,
        "loss_mode": args.loss_mode,
        "loss_tag": args.loss_tag,
        "fold_index": args.fold_index,
        "num_folds": args.num_folds,
        "epoch": epoch,
        "step": step,
        **values,
    }


def main() -> None:
    args = parse_args()
    if args.validate_every_epochs <= 0:
        raise ValueError("--validate-every-epochs must be at least 1")
    if args.max_val_steps == 0 or args.max_val_steps < -1:
        raise ValueError("--max-val-steps must be -1 or a positive integer")
    if args.stage1_vae_checkpoint is not None and args.stage1_scvi_checkpoint is not None:
        raise ValueError("--stage1-vae-checkpoint and --stage1-scvi-checkpoint are mutually exclusive")
    if args.stage1_vae_checkpoint is not None:
        args.representation_type = "gaussian_vae"
        # The Gaussian VAE is always trained on log1p_10k, so adopt it when the
        # user said nothing and reject only an explicit contradiction.
        if args.expression_transform is None:
            args.expression_transform = "log1p_10k"
        elif args.expression_transform != "log1p_10k":
            raise ValueError("A Stage-1 Gaussian VAE requires --expression-transform log1p_10k")
        if args.expression_loss_weight > 0 or args.source_recon_loss_weight > 0:
            raise ValueError("Stage-2 training must keep the pretrained VAE frozen")
    if args.stage1_scvi_checkpoint is not None:
        args.representation_type = "scvi"
        # The required transform depends on how Stage 1 was trained: raw counts
        # for a count likelihood, log1p_10k for a Normal likelihood. Adopted
        # from the checkpoint below, once it has been read.
        if args.expression_loss_weight > 0 or args.source_recon_loss_weight > 0:
            raise ValueError("Stage-2 training must keep the pretrained scVI module frozen")
    if args.stage1_scvi_checkpoint is None and args.latent_dim is None:
        args.latent_dim = 128
    if args.stage1_scvi_checkpoint is None and args.expression_transform is None:
        # No checkpoint to adopt from (legacy_mlp, or the Gaussian VAE branch
        # already resolved it): fall back to the historical default.
        args.expression_transform = "none"
    if args.checkpoint_dir is None:
        variant_part = f"_{args.variant}" if args.variant != "default" else ""
        args.checkpoint_dir = Path(f"checkpoints_level{args.level}{variant_part}_fold{args.fold_index}")

    torch.manual_seed(args.seed)
    device = choose_device(args.device)

    # Take the batch key and technology filter from the Stage-1 checkpoint
    # rather than from separate flags, so Stage 2 cannot silently train on a
    # different cell population -- or with different batch labels -- than the
    # frozen encoder was fitted on.
    stage1_batch_key = None
    stage1_keep_technologies = None
    stage1_batch_categories = None
    stage1_gene_ids = None
    if args.stage1_scvi_checkpoint is not None:
        from scvi_stage1_representation import peek_checkpoint_metadata

        stage1_meta = peek_checkpoint_metadata(args.stage1_scvi_checkpoint)
        stage1_batch_key = stage1_meta["batch_key"]
        stage1_keep_technologies = stage1_meta["keep_technologies"]
        print("stage1_batch_key:", stage1_batch_key)
        print("stage1_keep_technologies:", stage1_keep_technologies)
        print("stage1_latent_dim:", stage1_meta["latent_dim"])
        if args.latent_dim is None:
            args.latent_dim = stage1_meta["latent_dim"]
            print(f"latent_dim: adopted {args.latent_dim} from checkpoint")
        elif args.latent_dim != stage1_meta["latent_dim"]:
            raise ValueError(
                f"Stage 1 has latent_dim {stage1_meta['latent_dim']} but Stage 2 was given "
                f"{args.latent_dim}. The transition operates inside the frozen latent, so "
                "these must agree."
            )
        print("stage1_gene_likelihood:", stage1_meta["gene_likelihood"])
        print("stage1_expression_transform:", stage1_meta["expression_transform"])
        # Adopt the checkpoint's transform when the user did not specify one, so
        # the frozen encoder cannot silently be fed the wrong units. An explicit
        # contradicting value is still an error rather than being overridden.
        if args.expression_transform is None:
            args.expression_transform = stage1_meta["expression_transform"]
            print(f"expression_transform: adopted {args.expression_transform!r} from checkpoint")
        elif args.expression_transform != stage1_meta["expression_transform"]:
            raise ValueError(
                f"Stage 1 was trained with --expression-transform "
                f"{stage1_meta['expression_transform']!r} (gene_likelihood="
                f"{stage1_meta['gene_likelihood']!r}), but Stage 2 was given "
                f"{args.expression_transform!r}. Feeding the frozen encoder different "
                "units than it was fitted on produces a meaningless latent."
            )
        needs_lookup = stage1_keep_technologies is not None or stage1_batch_key == "technology"
        if needs_lookup and args.technology_lookup is None:
            raise ValueError(
                "The Stage-1 checkpoint was trained with a technology filter or "
                "--batch-key technology; pass the same --technology-lookup to Stage 2."
            )
        # The frozen encoder can only embed the batch categories it was fitted
        # on. Cells from a study whose patients all landed in Stage 1's held-out
        # fold have no trained embedding, so they are dropped by the loader
        # rather than failing mid-batch.
        mapping = stage1_meta["batch_category_to_index"]
        if mapping:
            stage1_batch_categories = sorted(mapping)
            print(f"stage1_batch_categories: {len(stage1_batch_categories)} known")
        # Subset to exactly the genes Stage 1 kept, in Stage 1's order.
        stage1_gene_ids = list(stage1_meta["gene_ids"])
        print(
            f"stage1_genes: {len(stage1_gene_ids)}"
            f" (min_cells_detected={stage1_meta['min_cells_detected']})"
        )

    common_loader_kwargs = dict(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        shuffle_rows=True,
        sample_by_probability=args.sample_by_probability,
        use_probability_as_weight=not args.no_probability_weight,
        time_delta_col=args.time_delta_col,
        default_time_delta=args.default_time_delta,
        expression_transform=args.expression_transform,
        group_column=args.group_column,
        num_workers=args.num_workers,
        seed=args.seed,
        batch_key=stage1_batch_key,
        technology_lookup_path=args.technology_lookup,
        keep_technologies=stage1_keep_technologies,
        allowed_batch_categories=stage1_batch_categories,
        keep_gene_ids=stage1_gene_ids,
    )

    train_dataset, train_loader = build_paired_h5ad_loader(
        **common_loader_kwargs,
        shuffle_shards=True,
        split="train" if args.num_folds > 1 else "all",
        num_folds=args.num_folds,
        fold_index=args.fold_index,
    )
    val_loader_kwargs = {
        **common_loader_kwargs,
        "shuffle_rows": False,
        "sample_by_probability": False,
    }
    val_dataset, val_loader = build_paired_h5ad_loader(
        **val_loader_kwargs,
        shuffle_shards=False,
        split="val" if args.num_folds > 1 else "all",
        num_folds=args.num_folds,
        fold_index=args.fold_index,
    )

    model = build_model(
        n_genes=train_dataset.n_genes,
        level=args.level,
        variant=args.variant,
        latent_dim=args.latent_dim,
        expression_hidden_dim=args.expression_hidden_dim,
        expression_depth=args.expression_depth,
        representation_type=args.representation_type,
        scvi_checkpoint=str(args.stage1_scvi_checkpoint) if args.stage1_scvi_checkpoint else None,
        model_dim=args.model_dim,
        depth=args.depth,
        heads=args.heads,
        num_tokens=args.num_tokens,
        dropout=args.dropout,
    ).to(device)

    if args.stage1_vae_checkpoint is not None:
        try:
            stage1 = torch.load(args.stage1_vae_checkpoint, map_location="cpu", weights_only=False)
        except TypeError:
            stage1 = torch.load(args.stage1_vae_checkpoint, map_location="cpu")
        if stage1.get("format") != "stage1_gaussian_vae_log1p10k_v1":
            raise ValueError("Unsupported Stage-1 VAE checkpoint format")
        if int(stage1["n_genes"]) != train_dataset.n_genes:
            raise ValueError("Stage-1 VAE and paired data have different gene counts")
        if list(stage1["gene_ids"]) != list(train_dataset.gene_ids):
            raise ValueError("Stage-1 VAE and paired data have different gene order")
        stage1_args = stage1.get("args", {})
        expected = {
            "latent_dim": args.latent_dim,
            "hidden_dim": args.expression_hidden_dim,
            "depth": args.expression_depth,
        }
        for key, value in expected.items():
            if int(stage1_args.get(key, value)) != int(value):
                raise ValueError(f"Stage-1 architecture mismatch for {key}")
        model.autoencoder.load_state_dict(stage1["vae_state_dict"], strict=True)
        model.autoencoder.requires_grad_(False)
        model.autoencoder.eval()

    if args.stage1_scvi_checkpoint is not None:
        # build_model() already loaded and shape-validated the scVI checkpoint
        # (n_genes, latent_dim) inside FiveLevelCellWorldModel.__init__; only
        # gene *order* and freezing remain to be done here.
        if list(model.autoencoder.checkpoint_metadata["gene_ids"]) != list(train_dataset.gene_ids):
            raise ValueError(
                "Stage-1 scVI checkpoint and the loaded data have different genes or gene "
                "order, even after subsetting to the checkpoint's gene list."
            )
        model.autoencoder.requires_grad_(False)
        model.autoencoder.eval()

    # Level 0 is the non-learned identity baseline: it has no transition
    # parameters by construction, so there is nothing to optimize. It runs a
    # single evaluation pass to produce comparable metrics.
    is_identity_baseline = args.level == 0
    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable_parameters and not is_identity_baseline:
        raise ValueError("No trainable Stage-2 parameters")
    if is_identity_baseline:
        if trainable_parameters:
            raise ValueError(
                "Level 0 must have no trainable parameters. Freeze the representation "
                "with --stage1-scvi-checkpoint or --stage1-vae-checkpoint."
            )
        if args.epochs != 1:
            print(f"level 0: forcing --epochs 1 (was {args.epochs}); nothing is trained")
            args.epochs = 1
    optimizer = (
        None
        if is_identity_baseline
        else torch.optim.AdamW(trainable_parameters, lr=args.lr, weight_decay=args.weight_decay)
    )
    chreode_config = ChreodeLossConfig(
        lambda_mmd=args.lambda_mmd,
        lambda_w2=args.lambda_w2,
        lambda_drift=args.lambda_drift,
        lambda_down=args.lambda_down,
        sinkhorn_epsilon=args.sinkhorn_epsilon,
        sinkhorn_iters=args.sinkhorn_iters,
    )

    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    # Select the checkpoint on the objective that is actually optimized, for
    # every loss mode.  This used to special-case CellFlow and select every
    # other mode on val_loss_pair, which silently broke --loss-mode chreode:
    # the run would optimize MMD/W2/drift but pick the epoch that best fit the
    # per-cell MSE, i.e. the noisy MOSCOT pairing target the distributional
    # objective exists to avoid.
    #
    # For --loss-mode pair_mse this is not a behaviour change: loss == loss_pair
    # there whenever the expression/latent/source-recon aux weights are 0
    # (their defaults), so previously-trained pair_mse checkpoints are still
    # selected identically.
    selection_metric = "val_loss_total"
    run_config = {
        **vars(args),
        "n_genes": train_dataset.n_genes,
        "train_batches_per_epoch": len(train_dataset),
        "val_batches_per_epoch": len(val_dataset),
        "selection_metric": selection_metric,
    }
    (args.checkpoint_dir / "run_config.json").write_text(json.dumps(run_config, indent=2, default=str), encoding="utf-8")

    fields = make_loss_fields()
    step_loss_path = args.checkpoint_dir / "step_losses.csv"
    train_epoch_path = args.checkpoint_dir / "train_epoch_losses.csv"
    val_epoch_path = args.checkpoint_dir / "val_epoch_losses.csv"

    print("device:", device)
    print("level:", args.level)
    print("variant:", args.variant)
    print("loss_mode:", args.loss_mode)
    print("loss_tag:", args.loss_tag)
    print("fold:", f"{args.fold_index}/{args.num_folds}")
    print("n_genes:", train_dataset.n_genes)
    print("train_batches_per_epoch:", len(train_dataset))
    print("val_batches_per_epoch:", len(val_dataset))
    print("checkpoint_dir:", args.checkpoint_dir)

    global_step = 0
    best_val_metric = math.inf
    start_time = time.time()

    for epoch in range(args.epochs):
        model.train()
        if is_identity_baseline:
            model.eval()
        if args.stage1_vae_checkpoint is not None:
            model.autoencoder.eval()
        train_stats = empty_stats()

        for batch in train_loader:
            if is_identity_baseline:
                with torch.no_grad():
                    losses = compute_losses(model, batch, device, args, chreode_config)
            else:
                losses = compute_losses(model, batch, device, args, chreode_config)

                optimizer.zero_grad(set_to_none=True)
                losses["loss"].backward()
                if args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(trainable_parameters, args.grad_clip)
                optimizer.step()

            add_stats(train_stats, losses)
            append_csv(
                step_loss_path,
                fields,
                log_row(
                    args,
                    "train",
                    epoch,
                    global_step,
                    {"n_steps": "", **{k: float(v.detach().cpu()) for k, v in losses.items()}},
                ),
            )

            if global_step % 10 == 0:
                elapsed = max(time.time() - start_time, 1e-6)
                print(
                    f"step={global_step} epoch={epoch} "
                    f"loss={losses['loss'].item():.6f} pair={losses['loss_pair'].item():.6f} "
                    f"mmd={losses['metric_mmd'].item():.6f} w2={losses['metric_w2'].item():.6f} "
                    f"drift={losses['metric_drift'].item():.6f} down={losses['metric_down'].item():.6f} "
                    f"steps_per_sec={global_step / elapsed:.3f}"
                )

            global_step += 1
            if args.save_every > 0 and global_step % args.save_every == 0:
                path = save_checkpoint(args.checkpoint_dir, model, optimizer, args, global_step, epoch, train_dataset)
                print("saved:", path)

            if args.max_steps > 0 and global_step >= args.max_steps:
                break

        train_row = log_row(args, "train", epoch, global_step, mean_stats(train_stats))
        append_csv(train_epoch_path, fields, train_row)

        reached_last_step = args.max_steps > 0 and global_step >= args.max_steps
        should_validate = (
            epoch == 0
            or (epoch + 1) % args.validate_every_epochs == 0
            or epoch == args.epochs - 1
            or reached_last_step
        )
        if should_validate:
            model.eval()
            val_stats = empty_stats()
            with torch.no_grad():
                for val_step, batch in enumerate(val_loader):
                    if args.max_val_steps > 0 and val_step >= args.max_val_steps:
                        break
                    losses = compute_losses(model, batch, device, args, chreode_config)
                    add_stats(val_stats, losses)

            val_row = log_row(args, "val", epoch, global_step, mean_stats(val_stats))
            append_csv(val_epoch_path, fields, val_row)

            print(
                f"epoch={epoch} "
                f"train_loss={train_row['loss']:.6f} train_pair={train_row['loss_pair']:.6f} "
                f"val_loss={val_row['loss']:.6f} val_pair={val_row['loss_pair']:.6f} "
                f"val_mmd={val_row['metric_mmd']:.6f} val_w2={val_row['metric_w2']:.6f}"
            )

            current_val_metric = val_row["loss"]
            if current_val_metric < best_val_metric:
                best_val_metric = current_val_metric
                path = save_checkpoint(
                    args.checkpoint_dir,
                    model,
                    optimizer,
                    args,
                    global_step,
                    epoch,
                    train_dataset,
                    name=(
                        f"best_model_level{args.level}"
                        f"{'_' + args.variant if args.variant != 'default' else ''}"
                        f"_fold{args.fold_index}.pt"
                    ),
                )
                print("saved_best:", path)
        else:
            print(
                f"epoch={epoch} train_loss={train_row['loss']:.6f} "
                f"train_pair={train_row['loss_pair']:.6f} validation=skipped"
            )

        if args.save_each_epoch:
            path = save_checkpoint(args.checkpoint_dir, model, optimizer, args, global_step, epoch, train_dataset)
            print("saved_epoch_checkpoint:", path)

        if args.max_steps > 0 and global_step >= args.max_steps:
            break

    path = save_checkpoint(args.checkpoint_dir, model, optimizer, args, global_step, args.epochs - 1, train_dataset)
    print("saved_final:", path)
    print("selection_metric:", selection_metric)
    print("best_val_selection_metric:", best_val_metric)


if __name__ == "__main__":
    main()
