#!/usr/bin/env python3
"""
Frozen scVI encoder/decoder wrapper used as the Stage-1 representation.

This module is imported lazily by five_level_cell_world_model.py so that
`scvi-tools` (a heavy dependency: torch, lightning, jax/flax, etc.) is only
required when --representation-type scvi is actually requested.

Design decisions, made explicit here because they differ from a "vanilla"
scvi-tools workflow:

1. Batch conditioning is on dataset_id, not patient_id.
   The training data pools scRNA-seq from multiple source studies/platforms
   (see docs/stage1_scvi_notes.md), so dataset_id is a real technical batch
   covariate scVI should correct for. patient_id is instead the grouping
   variable for cross-validation folds (see compute_group_to_fold() in
   sample_paired_h5ad_dataloader.py) -- using the same variable for both
   would mean held-out CV patients are, by construction, unseen batch
   categories. Encode/decode both take an explicit batch_categories list
   (dataset_id strings) so this class never has to guess.

   If a checkpoint was trained with only a single (dummy) batch category
   (--batch-key none in train_stage1_scvi.py, or an old checkpoint predating
   this feature), batch_categories is ignored entirely and every cell is
   mapped to index 0 -- this keeps old, no-batch-key checkpoints loadable.

2. decode(z) returns log1p(px_scale * reference_library_size), not raw
   counts and not px_rate for an observed library. px_scale (scVI's
   library-size-independent normalized expression, sums to 1 per cell) is
   decoded against a *fixed* reference library size (default 1e4), matching
   the log1p_10k units used everywhere else in this project
   (transform_expression_rows() in sample_paired_h5ad_dataloader.py). This
   is the same trick scvi.model.SCVI.get_normalized_expression() uses
   internally.

3. encode(x) requires raw counts (--expression-transform none), not
   log1p_10k. scVI applies its own internal log1p before the encoder
   (log_variational=True) and its NB/ZINB likelihood is defined on counts.
   Feeding it log1p_10k-transformed data would double-transform.

4. No scvi.model.SCVI.save()/load(). The checkpoint (stage1_scvi_v1) stores
   only the underlying scvi.module.VAE state_dict() plus its plain
   constructor kwargs and the batch category-to-index mapping, not a full
   SCVI model bundle with its AnnData registry. Stage 2 rebuilds the module
   directly (VAE(**module_init_kwargs) + load_state_dict) -- no AnnData
   needs to travel with the checkpoint.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import torch
from torch import nn

CHECKPOINT_FORMAT = "stage1_scvi_v1"
_SINGLE_BATCH_KEY = "__single_batch__"


def peek_checkpoint_metadata(checkpoint_path: str | Path) -> Dict[str, Any]:
    """
    Read a Stage-1 scVI checkpoint's metadata without constructing the module.

    Stage 2 needs the batch_key and technology filter that Stage 1 was trained
    with *before* it builds its dataloaders, so that the loader emits batch
    labels drawn from exactly the categories the encoder knows. Taking these
    from the checkpoint instead of from separately-passed flags makes it
    impossible for the two stages to silently disagree.
    """
    try:
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(checkpoint_path, map_location="cpu")
    if payload.get("format") != CHECKPOINT_FORMAT:
        raise ValueError(
            f"Unsupported Stage-1 scVI checkpoint format: {payload.get('format')!r}"
        )
    return {
        "batch_key": payload.get("batch_key"),
        "batch_category_to_index": payload.get("batch_category_to_index"),
        "keep_technologies": payload.get("keep_technologies"),
        # Older checkpoints predate these fields; they were all raw-count/zinb.
        "expression_transform": payload.get("expression_transform", "none"),
        "gene_likelihood": payload.get("gene_likelihood", "zinb"),
        "min_cells_detected": payload.get("min_cells_detected", 0),
        # Part of the contract: Stage 2's transition operates in this latent, so
        # it should adopt the dimension rather than be told it separately.
        "latent_dim": int(payload["module_init_kwargs"]["n_latent"]),
        "n_genes": payload["n_genes"],
        # The genes the encoder was actually fitted on, in order. Stage 2 must
        # subset its columns to exactly this list; the shards still hold the
        # full unified panel.
        "gene_ids": payload["gene_ids"],
    }


def _load_vae_module_class():
    # Lazy import: scvi-tools is only required for this representation type.
    from scvi.module import VAE

    return VAE


class ScVIExpressionAutoencoder(nn.Module):
    """Frozen scVI (negative-binomial) encoder/decoder for Stage 2 use."""

    def __init__(
        self,
        module_init_kwargs: Dict[str, Any],
        module_state_dict: Dict[str, torch.Tensor],
        batch_category_to_index: Optional[Dict[str, int]] = None,
        reference_library_size: float = 1e4,
        mean_log_library: Optional[float] = None,
    ):
        super().__init__()
        vae_cls = _load_vae_module_class()
        self.module_init_kwargs = dict(module_init_kwargs)
        self.vae_module = vae_cls(**self.module_init_kwargs)
        self.vae_module.load_state_dict(module_state_dict, strict=True)
        self.reference_library_size = float(reference_library_size)
        self.n_latent = int(self.module_init_kwargs["n_latent"])
        self.gene_likelihood = str(self.module_init_kwargs.get("gene_likelihood", "zinb"))
        if self.gene_likelihood == "normal" and mean_log_library is None:
            raise ValueError(
                "gene_likelihood='normal' requires mean_log_library: its px.loc is "
                "library-scaled, so decoding needs the scale the model was fitted at."
            )
        self.mean_log_library = mean_log_library
        self.batch_category_to_index: Dict[str, int] = dict(
            batch_category_to_index or {_SINGLE_BATCH_KEY: 0}
        )

    def _batch_index(
        self,
        batch_categories: Optional[Iterable[str]],
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        if len(self.batch_category_to_index) <= 1:
            # No real batch correction was trained (dummy single category,
            # or an old checkpoint predating batch conditioning) -- every
            # cell maps to index 0 regardless of what's passed in.
            return torch.zeros(batch_size, 1, dtype=torch.long, device=device)
        if batch_categories is None:
            raise ValueError(
                "This scVI checkpoint was trained with real batch categories "
                f"({sorted(self.batch_category_to_index)}); batch_categories "
                "must be provided to encode()/decode()."
            )
        categories = list(batch_categories)
        if len(categories) != batch_size:
            raise ValueError(
                f"batch_categories has {len(categories)} entries but batch_size is {batch_size}"
            )
        try:
            indices = [self.batch_category_to_index[str(c)] for c in categories]
        except KeyError as exc:
            raise ValueError(
                f"Unseen batch category {exc.args[0]!r}; known categories: "
                f"{sorted(self.batch_category_to_index)}"
            ) from exc
        return torch.tensor(indices, dtype=torch.long, device=device).view(-1, 1)

    def encode(
        self, x: torch.Tensor, batch_categories: Optional[Iterable[str]] = None
    ) -> torch.Tensor:
        """Deterministic posterior mean, matching GaussianExpressionVAE.encode."""
        batch_index = self._batch_index(batch_categories, x.shape[0], x.device)
        inference_out = self.vae_module.inference(x, batch_index)
        qz = inference_out["qz"]
        return qz.loc

    def decode(
        self, z: torch.Tensor, batch_categories: Optional[Iterable[str]] = None
    ) -> torch.Tensor:
        """
        Decode a latent back to log1p-scale expression.

        The decoding rule depends on the generative likelihood, and the two
        cases are NOT interchangeable:

        count likelihoods (nb / zinb / poisson)
            `px.scale` is scVI's library-size-independent normalized expression
            (sums to 1 across genes). Multiplying by a fixed reference library
            size and taking log1p puts it in the same units as this project's
            `log1p_10k` transform.

        normal likelihood
            The model was trained on already-log1p-normalized input, so the
            decoded mean `px.loc` is *already* in log1p space and must be
            returned as-is. Note `px.scale` also exists on a Normal but means
            the standard deviation -- applying the count-likelihood formula to
            it would silently return log1p(stddev * library), which is
            meaningless. Hence the explicit branch rather than a hasattr check.
        """
        batch_index = self._batch_index(batch_categories, z.shape[0], z.device)
        log_library = (
            self.mean_log_library
            if self.gene_likelihood == "normal"
            else math.log(self.reference_library_size)
        )
        library = torch.full(
            (z.shape[0], 1), float(log_library), dtype=z.dtype, device=z.device
        )
        generative_out = self.vae_module.generative(z, library, batch_index)
        px = generative_out["px"]

        if self.gene_likelihood == "normal":
            return px.loc

        px_scale = getattr(px, "scale", None)
        if px_scale is None:
            px_scale = generative_out["px_scale"]
        return torch.log1p(px_scale * self.reference_library_size)

    def forward(
        self, x: torch.Tensor, batch_categories: Optional[Iterable[str]] = None
    ) -> torch.Tensor:
        return self.decode(self.encode(x, batch_categories), batch_categories)

    @classmethod
    def from_checkpoint(cls, checkpoint_path: str | Path) -> "ScVIExpressionAutoencoder":
        try:
            payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        except TypeError:
            payload = torch.load(checkpoint_path, map_location="cpu")
        if payload.get("format") != CHECKPOINT_FORMAT:
            raise ValueError(
                f"Unsupported Stage-1 scVI checkpoint format: {payload.get('format')!r}"
            )
        model = cls(
            module_init_kwargs=payload["module_init_kwargs"],
            module_state_dict=payload["module_state_dict"],
            # .get() so checkpoints saved before batch conditioning was added
            # still load (as single-dummy-batch, i.e. no batch correction).
            batch_category_to_index=payload.get("batch_category_to_index"),
            reference_library_size=payload.get("reference_library_size", 1e4),
            mean_log_library=payload.get("mean_log_library"),
        )
        model.checkpoint_metadata = {
            "n_genes": payload["n_genes"],
            "gene_ids": payload["gene_ids"],
            "gene_symbols": payload["gene_symbols"],
            "batch_key": payload.get("batch_key"),
            "fold_index": payload.get("fold_index"),
            "num_folds": payload.get("num_folds"),
            "group_column": payload.get("group_column"),
            "scvi_version": payload.get("scvi_version"),
            "val_reconstruction_error": payload.get("val_reconstruction_error"),
        }
        return model
