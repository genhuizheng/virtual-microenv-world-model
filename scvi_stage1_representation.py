#!/usr/bin/env python3
"""
Frozen scVI encoder/decoder wrapper used as the Stage-1 representation.

This module is imported lazily by five_level_cell_world_model.py so that
`scvi-tools` (a heavy dependency: torch, lightning, jax/flax, etc.) is only
required when --representation-type scvi is actually requested.

Design decisions, made explicit here because they differ from a "vanilla"
scvi-tools workflow:

1. No batch_key / batch correction.
   Threading a per-cell batch/patient covariate through decode() would
   require every call site in five_level_cell_world_model.py and
   train_five_level_world_model.py (encode(x), decode(z), checkpoint
   save/load, ...) to also carry batch metadata for arbitrary *predicted*
   latents, not just observed cells. scVI is therefore trained with a single
   dummy batch category (n_batch=1). Batch correction across dataset_id can
   be added later without changing this file's public encode()/decode()
   signatures.

2. decode(z) returns log1p(px_scale * reference_library_size), not raw
   counts and not px_rate for an observed library size. This keeps the
   decoder's output in the same "log1p_10k" units used everywhere else in
   this project (see transform_expression_rows() in
   sample_paired_h5ad_dataloader.py), by fixing the decoder's library-size
   input to a constant reference_library_size (default 1e4) instead of a
   per-cell observed library. px_scale is library-size independent (it sums
   to 1 across genes per cell), so this is a deterministic, well-defined
   choice, analogous to scvi.model.SCVI.get_normalized_expression().

3. encode(x) requires raw counts, not log1p_10k expression. scVI applies its
   own internal log1p transform before the encoder (log_variational=True).
   Feeding it already-log1p_10k-transformed data would double-transform and
   also breaks the NB/ZINB reconstruction likelihood, which is defined on
   counts. Train/Stage-2 scripts must use --expression-transform none when
   --stage1-scvi-checkpoint is set.

4. This wrapper never goes through scvi.model.SCVI.load()/save(). It only
   keeps the underlying torch module (scvi.module.VAE) state_dict plus the
   plain-dict constructor kwargs needed to rebuild it. This avoids bundling
   or re-validating a full AnnData object at Stage-2 load time (which may
   run on a different machine / TACC node than Stage-1 training), mirroring
   the plain torch.save()/load_state_dict() pattern already used for the
   Gaussian VAE Stage-1 checkpoint.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict

import torch
from torch import nn

CHECKPOINT_FORMAT = "stage1_scvi_v1"


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
        reference_library_size: float = 1e4,
    ):
        super().__init__()
        vae_cls = _load_vae_module_class()
        self.module_init_kwargs = dict(module_init_kwargs)
        self.vae_module = vae_cls(**self.module_init_kwargs)
        self.vae_module.load_state_dict(module_state_dict, strict=True)
        self.reference_library_size = float(reference_library_size)
        self.n_latent = int(self.module_init_kwargs["n_latent"])

    @staticmethod
    def _dummy_batch_index(batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(batch_size, 1, dtype=torch.long, device=device)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Deterministic posterior mean, matching GaussianExpressionVAE.encode."""
        batch_index = self._dummy_batch_index(x.shape[0], x.device)
        inference_out = self.vae_module.inference(x, batch_index)
        qz = inference_out["qz"]
        return qz.loc

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """log1p(px_scale * reference_library_size); see module docstring."""
        batch_index = self._dummy_batch_index(z.shape[0], z.device)
        library = torch.full(
            (z.shape[0], 1),
            math.log(self.reference_library_size),
            dtype=z.dtype,
            device=z.device,
        )
        generative_out = self.vae_module.generative(z, library, batch_index)
        px = generative_out["px"]
        px_scale = px.scale if hasattr(px, "scale") else generative_out["px_scale"]
        return torch.log1p(px_scale * self.reference_library_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))

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
            reference_library_size=payload.get("reference_library_size", 1e4),
        )
        model.checkpoint_metadata = {
            "n_genes": payload["n_genes"],
            "gene_ids": payload["gene_ids"],
            "gene_symbols": payload["gene_symbols"],
            "fold_index": payload.get("fold_index"),
            "num_folds": payload.get("num_folds"),
            "group_column": payload.get("group_column"),
            "scvi_version": payload.get("scvi_version"),
            "val_reconstruction_error": payload.get("val_reconstruction_error"),
        }
        return model
