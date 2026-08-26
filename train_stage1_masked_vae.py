#!/usr/bin/env python3
"""Train a Stage-1 masked-reconstruction VAE.

The alternative to scVI for this data. scVI has no per-cell gene mask, so it
cannot distinguish a gene a dataset never measured from one the cell does not
express; both are stored as zero in the unified panel. This script scores the
reconstruction only over genes each cell's dataset actually measured.

Masks come from the gene_masks_by_dataset.npz built by scanning the zero-fill
resolution maps: a [n_datasets x n_panel_genes] boolean matrix plus the
dataset_id and panel-gene orders. Rows are matched by dataset_id.

Compared with train_stage1_scvi.py this keeps everything else identical -- the
same patient-grouped folds, the same technology filter, the same gene panel
options, the same batch key derived from source_study -- so a checkpoint from
either script is a drop-in Stage-1 representation and the two are directly
comparable through evaluate_stage1_scvi.py.

Usage:
    python train_stage1_masked_vae.py \\
        --data-dir paired_training_h5ad_50k \\
        --checkpoint-dir stage1_masked \\
        --gene-mask-npz ../Virtual_microenv/gene_masks_by_dataset.npz \\
        --keep-gene-list ../Virtual_microenv/genes_measured_50pct.txt \\
        --batch-key source_study --num-folds 5 --fold-index 0 \\
        --technology-lookup ../Virtual_microenv/dataset_platform_lookup.tsv \\
        --keep-technologies "10x genomics" --device cuda --overwrite
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from masked_vae import (
    CHECKPOINT_FORMAT,
    MaskedGaussianVAE,
    adversarial_loss,
    masked_vae_loss,
)
from sample_paired_h5ad_dataloader import safe_get_gene_symbols
from train_stage1_scvi import build_split_anndata


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--checkpoint-dir", type=Path, required=True)
    p.add_argument(
        "--gene-mask-npz",
        type=Path,
        default=None,
        help="npz with dataset_ids / masks / panel, as written by the zero-fill "
        "resolution-map scan. Omit to train WITHOUT masking, which is the control "
        "condition rather than the intended use.",
    )
    p.add_argument("--max-epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-2)
    p.add_argument("--kl-weight", type=float, default=1e-3)
    p.add_argument("--latent-dim", type=int, default=128)
    p.add_argument("--n-hidden", type=int, default=512)
    p.add_argument("--n-layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--batch-embed-dim", type=int, default=16)
    p.add_argument(
        "--adversarial-weight",
        type=float,
        default=0.0,
        help="Weight on the gradient-reversed study classifier. 0 disables it, which "
        "is the control condition. When > 0 the batch key is used ONLY to supply "
        "training labels for the adversary -- the encoder and decoder are not "
        "conditioned on it, so inference needs no batch label and an unseen study is "
        "not a problem. Mutually exclusive with batch conditioning.",
    )
    p.add_argument(
        "--adversarial-warmup-epochs",
        type=int,
        default=10,
        help="Ramp the adversarial weight linearly from 0 over this many epochs. "
        "Applying full adversarial pressure to an untrained encoder tends to collapse "
        "the latent before it has learned anything worth keeping.",
    )
    p.add_argument(
        "--expression-transform",
        type=str,
        default="log1p_10k",
        help="log1p_10k (default) or bin_<N>. Raw counts are not appropriate for an "
        "MSE reconstruction.",
    )
    p.add_argument("--num-folds", type=int, default=5)
    p.add_argument("--fold-index", type=int, default=0)
    p.add_argument("--group-column", type=str, default="patient_id")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--batch-key",
        choices=["none", "source_study", "technology", "dataset_id", "patient_id"],
        default="source_study",
    )
    p.add_argument("--technology-lookup", type=Path, default=None)
    p.add_argument("--keep-technologies", type=str, default=None)
    p.add_argument("--keep-gene-list", type=Path, default=None)
    p.add_argument("--min-cells-detected", type=int, default=0)
    p.add_argument("--no-probability-weight", action="store_true")
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def choose_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def load_gene_masks(path: Path, gene_ids: list[str]) -> tuple[dict[str, int], np.ndarray]:
    """
    Load dataset masks and align their columns to the training gene order.

    The npz is stored against the full unified panel; the training panel is
    usually a subset in a specific order, so columns are reindexed rather than
    assumed to match.
    """
    data = np.load(path, allow_pickle=True)
    dataset_ids = [str(d) for d in data["dataset_ids"]]
    masks = np.asarray(data["masks"], dtype=bool)
    panel = [str(g) for g in data["panel"]]

    position = {g: i for i, g in enumerate(panel)}
    missing = [g for g in gene_ids if g not in position]
    if missing:
        raise ValueError(
            f"{len(missing)} training gene(s) are absent from the mask panel, e.g. "
            f"{missing[:5]}. The mask npz must cover the panel being trained on."
        )
    columns = np.asarray([position[g] for g in gene_ids], dtype=np.int64)
    aligned = masks[:, columns]
    index = {ds: i for i, ds in enumerate(dataset_ids)}
    return index, aligned


def build_mask_matrix(
    obs_dataset_ids: np.ndarray,
    mask_index: dict[str, int],
    mask_matrix: np.ndarray,
) -> np.ndarray:
    rows = []
    unknown = set()
    for ds in obs_dataset_ids:
        key = str(ds).strip()
        if key not in mask_index:
            unknown.add(key)
            continue
        rows.append(mask_index[key])
    if unknown:
        raise ValueError(
            f"{len(unknown)} dataset_id(s) have no mask, e.g. {sorted(unknown)[:5]}. "
            "Training on a partial mask would silently score unmeasured genes."
        )
    return mask_matrix[np.asarray(rows, dtype=np.int64)]


def run_epoch(
    model: MaskedGaussianVAE,
    x_all: torch.Tensor,
    mask_all: torch.Tensor | None,
    batch_all: torch.Tensor | None,
    weight_all: torch.Tensor | None,
    optimizer: torch.optim.Optimizer | None,
    batch_size: int,
    kl_weight: float,
    device: torch.device,
    generator: torch.Generator,
    study_all: torch.Tensor | None = None,
    adversarial_weight: float = 0.0,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    n = x_all.shape[0]
    order = (
        torch.randperm(n, generator=generator) if training else torch.arange(n)
    )
    totals = np.zeros(6, dtype=np.float64)
    steps = 0
    context = torch.enable_grad if training else torch.no_grad
    with context():
        for start in range(0, n, batch_size):
            idx = order[start : start + batch_size]
            x = x_all[idx].to(device, non_blocking=True)
            mask = mask_all[idx].to(device) if mask_all is not None else None
            bidx = batch_all[idx].to(device) if batch_all is not None else None
            w = weight_all[idx].to(device) if weight_all is not None else None

            out = model(x, bidx)
            losses = masked_vae_loss(out, x, mask, w, kl_weight)
            total = losses["loss"]

            adv = torch.zeros((), device=device)
            adv_acc = torch.zeros((), device=device)
            if adversarial_weight > 0 and study_all is not None:
                # The gradient reversal is inside adversarial_logits, so ADDING
                # this term trains the classifier to predict study while pushing
                # the encoder to make study unpredictable.
                parts = adversarial_loss(
                    model, out["mu"], study_all[idx].to(device), adversarial_weight
                )
                adv, adv_acc = parts["adversarial"], parts["adversarial_accuracy"]
                total = total + adv

            if not torch.isfinite(total):
                raise FloatingPointError("Non-finite masked-VAE loss")
            if training:
                optimizer.zero_grad(set_to_none=True)
                total.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            totals += [
                float(total.detach()),
                float(losses["reconstruction"].detach()),
                float(losses["kl"].detach()),
                float(losses["mean_measured_genes"].detach()),
                float(adv.detach()),
                float(adv_acc.detach()),
            ]
            steps += 1
    if steps == 0:
        raise ValueError("no batches produced")
    mean = totals / steps
    return {
        "steps": steps,
        "loss": mean[0],
        "reconstruction": mean[1],
        "kl": mean[2],
        "mean_measured_genes": mean[3],
        "adversarial": mean[4],
        "adversarial_accuracy": mean[5],
    }


def to_dense_tensor(matrix) -> torch.Tensor:
    dense = matrix.toarray() if hasattr(matrix, "toarray") else np.asarray(matrix)
    return torch.as_tensor(np.asarray(dense, dtype=np.float32))


def main() -> None:
    args = parse_args()
    if not 0 <= args.fold_index < args.num_folds:
        raise ValueError("--fold-index must be in [0, --num-folds)")
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_path = args.checkpoint_dir / f"best_stage1_masked_vae_fold{args.fold_index}.pt"
    history_path = args.checkpoint_dir / f"masked_vae_fold{args.fold_index}_history.csv"
    if not args.overwrite and best_path.exists():
        raise FileExistsError("outputs already exist; use --overwrite")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = choose_device(args.device)
    generator = torch.Generator().manual_seed(args.seed)

    keep_technologies = None
    if args.keep_technologies:
        from sample_paired_h5ad_dataloader import normalize_technology

        keep_technologies = {
            normalize_technology(t) for t in args.keep_technologies.split(",") if t.strip()
        }

    keep_gene_ids = None
    if args.keep_gene_list is not None:
        rows = [r.strip() for r in args.keep_gene_list.read_text().splitlines() if r.strip()]
        if rows and rows[0].lower().split("\t")[0] in {"gene_id", "gene"}:
            rows = rows[1:]
        keep_gene_ids = [r.split("\t")[0] for r in rows]
        print(f"keep_gene_list: {len(keep_gene_ids)} genes")

    train_adata, val_adata = build_split_anndata(
        args.data_dir,
        args.group_column,
        args.num_folds,
        args.fold_index,
        args.seed,
        technology_lookup_path=args.technology_lookup,
        keep_technologies=keep_technologies,
        expression_transform=args.expression_transform,
        min_cells_detected=args.min_cells_detected,
        keep_gene_ids=keep_gene_ids,
    )
    gene_ids = list(train_adata.var_names.astype(str))
    print(f"train_cells={train_adata.n_obs} val_cells={val_adata.n_obs} n_genes={len(gene_ids)}")

    mask_index = mask_matrix = None
    if args.gene_mask_npz is not None:
        mask_index, mask_matrix = load_gene_masks(args.gene_mask_npz, gene_ids)
        coverage = mask_matrix.mean()
        print(
            f"gene masks: {mask_matrix.shape[0]} datasets, mean coverage {coverage:.3f} "
            f"of the {len(gene_ids)}-gene panel"
        )
    else:
        print("WARNING: no --gene-mask-npz; training WITHOUT masking (control condition)")

    if args.adversarial_weight > 0 and args.batch_key == "none":
        raise ValueError(
            "--adversarial-weight needs --batch-key to supply training labels for the "
            "study classifier. The encoder/decoder are NOT conditioned on it; the labels "
            "are used only to train the adversary."
        )

    batch_key = None if args.batch_key == "none" else args.batch_key
    batch_category_to_index: dict[str, int] | None = None
    train_batch = val_batch = None
    if batch_key is not None:
        categories = sorted(set(train_adata.obs[batch_key].astype(str)))
        batch_category_to_index = {c: i for i, c in enumerate(categories)}
        print(f"batch_key={batch_key} ({len(categories)} categories in train split)")
        unseen = set(val_adata.obs[batch_key].astype(str)) - set(categories)
        if unseen and val_adata.n_obs:
            keep = val_adata.obs[batch_key].astype(str).isin(set(categories)).to_numpy()
            dropped = int((~keep).sum())
            print(
                f"WARNING: {len(unseen)} {batch_key} value(s) only in validation; dropping "
                f"{dropped}/{val_adata.n_obs} cells ({100.0*dropped/val_adata.n_obs:.2f}%)"
            )
            val_adata = val_adata[keep].copy()
        train_batch = torch.as_tensor(
            [batch_category_to_index[c] for c in train_adata.obs[batch_key].astype(str)],
            dtype=torch.long,
        )
        if val_adata.n_obs:
            val_batch = torch.as_tensor(
                [batch_category_to_index[c] for c in val_adata.obs[batch_key].astype(str)],
                dtype=torch.long,
            )

    def masks_for(adata) -> torch.Tensor | None:
        if mask_index is None or adata.n_obs == 0:
            return None
        m = build_mask_matrix(adata.obs["dataset_id"].to_numpy(), mask_index, mask_matrix)
        return torch.as_tensor(m.astype(np.float32))

    x_train = to_dense_tensor(train_adata.X)
    x_val = to_dense_tensor(val_adata.X) if val_adata.n_obs else None
    m_train, m_val = masks_for(train_adata), masks_for(val_adata)

    w_train = w_val = None
    if not args.no_probability_weight and "probability" in train_adata.obs.columns:
        w_train = torch.as_tensor(
            train_adata.obs["probability"].to_numpy(dtype=np.float32)
        )
        if val_adata.n_obs:
            w_val = torch.as_tensor(val_adata.obs["probability"].to_numpy(dtype=np.float32))

    # Built once and reused verbatim in the checkpoint, so the saved
    # architecture always matches what was actually constructed. Deriving these
    # again at save time is how an adversarial model ends up recorded as a
    # conditioned one and fails to reload.
    module_init_kwargs = {
        "n_genes": len(gene_ids),
        "latent_dim": args.latent_dim,
        "hidden_dim": args.n_hidden,
        "n_layers": args.n_layers,
        "dropout": args.dropout,
        # Adversarial invariance and batch conditioning are alternatives. With
        # the adversary on, the model carries no batch embedding at all, which
        # is what makes inference label-free on an unseen study.
        "n_batches": 1 if args.adversarial_weight > 0
        else (len(batch_category_to_index) if batch_category_to_index else 1),
        "batch_embed_dim": args.batch_embed_dim,
        "n_adversarial_classes": (
            len(batch_category_to_index)
            if args.adversarial_weight > 0 and batch_category_to_index
            else 0
        ),
    }
    model = MaskedGaussianVAE(**module_init_kwargs).to(device)
    if args.adversarial_weight > 0:
        print(
            f"adversarial: weight={args.adversarial_weight} over "
            f"{len(batch_category_to_index)} studies, warmup={args.adversarial_warmup_epochs} epochs "
            f"(encoder/decoder NOT conditioned; inference needs no batch label)"
        )
        chance = 1.0 / max(len(batch_category_to_index), 1)
        print(f"  adversary accuracy should fall toward chance = {chance:.4f}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # With the adversary on, the model has no batch embedding, so no batch index
    # is fed to encoder/decoder -- the labels go to the classifier instead.
    cond_train = None if args.adversarial_weight > 0 else train_batch
    cond_val = None if args.adversarial_weight > 0 else val_batch
    study_train = train_batch if args.adversarial_weight > 0 else None
    study_val = val_batch if args.adversarial_weight > 0 else None

    rows, best = [], float("inf")
    for epoch in range(args.max_epochs):
        # Linear warmup: full adversarial pressure on an untrained encoder tends
        # to collapse the latent before it has learned anything worth keeping.
        if args.adversarial_warmup_epochs > 0:
            ramp = min(1.0, (epoch + 1) / args.adversarial_warmup_epochs)
        else:
            ramp = 1.0
        adv_w = args.adversarial_weight * ramp

        tr = run_epoch(model, x_train, m_train, cond_train, w_train, optimizer,
                       args.batch_size, args.kl_weight, device, generator,
                       study_all=study_train, adversarial_weight=adv_w)
        if x_val is not None:
            va = run_epoch(model, x_val, m_val, cond_val, w_val, None,
                           args.batch_size, args.kl_weight, device, generator,
                           study_all=study_val, adversarial_weight=adv_w)
        else:
            va = {k: float("nan") for k in tr}
        rows.append({"epoch": epoch, **{f"train_{k}": v for k, v in tr.items()},
                     **{f"val_{k}": v for k, v in va.items()}})
        if epoch % 5 == 0 or epoch == args.max_epochs - 1:
            msg = (f"epoch={epoch} train={tr['loss']:.6g} val={va['loss']:.6g} "
                   f"recon={va['reconstruction']:.6g} measured={tr['mean_measured_genes']:.0f}")
            if args.adversarial_weight > 0:
                msg += (f" adv_w={adv_w:.3f} adv_acc={tr['adversarial_accuracy']:.4f}")
            print(msg)

        score = va["loss"] if x_val is not None else tr["loss"]
        if score < best:
            best = score
            torch.save(
                {
                    "format": CHECKPOINT_FORMAT,
                    "module_state_dict": model.state_dict(),
                    "module_init_kwargs": module_init_kwargs,
                    "batch_key": batch_key,
                    "batch_category_to_index": batch_category_to_index,
                    "expression_transform": args.expression_transform,
                    "min_cells_detected": args.min_cells_detected,
                    "keep_technologies": sorted(keep_technologies) if keep_technologies else None,
                    "masked": mask_index is not None,
                    "adversarial_weight": args.adversarial_weight,
                    "n_genes": len(gene_ids),
                    "gene_ids": gene_ids,
                    "gene_symbols": safe_get_gene_symbols(train_adata.var, gene_ids),
                    "epoch": epoch,
                    "args": vars(args),
                },
                best_path,
            )

    with history_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (args.checkpoint_dir / f"masked_vae_fold{args.fold_index}_config.json").write_text(
        json.dumps(vars(args), indent=2, default=str) + "\n", encoding="utf-8"
    )
    print(f"best_checkpoint={best_path} best_val={best:.6g}")


if __name__ == "__main__":
    main()
