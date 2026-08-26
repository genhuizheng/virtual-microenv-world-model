#!/usr/bin/env python3
"""Batch-conditioned Gaussian VAE with masked reconstruction.

Why this exists alongside scVI
------------------------------
The unified gene panel is a zero-filled union across source studies, so a zero
means one of two very different things:

    biological zero   the gene was measured, this cell does not express it
    structural zero   the gene was never in that dataset's panel

scVI cannot tell them apart -- it has no per-cell gene mask -- so it learns
"gene X is off" from cells where X was never looked at. Worse, the *pattern* of
structural zeros identifies the source study, so a study fingerprint is baked
into the expression values themselves, where no batch covariate can reach it.

Measured on this project's data: a dataset measures 10,972-40,581 of the 46,143
panel genes (median 25,259), so roughly 45% of a typical cell's vector is
padding rather than biology.

This module fixes that directly: the reconstruction loss is evaluated only over
genes the cell's dataset actually measured. Unmeasured genes contribute nothing
to the loss and nothing to the gradient.

Design notes
------------
1. Gaussian likelihood on log1p input, with a plain MSE reconstruction and no
   learned per-gene variance. That is deliberate: scVI's Normal likelihood
   drove per-gene variance toward zero on all-zero columns (23.9% of genes
   collapsed below 1e-3, training loss went negative and held-out error
   exploded). An MSE term has no variance parameter to collapse.

2. Per-cell loss normalization divides by that cell's mask sum, not by the gene
   count. Cells from wide-panel datasets would otherwise dominate the gradient
   purely for having been measured more thoroughly -- reintroducing a
   study-level bias through the back door.

3. Batch conditioning concatenates a learned embedding to both encoder input
   and decoder input (the conditional-VAE form). The decoder is conditioned so
   the latent does not have to carry batch identity; the encoder is conditioned
   so it can subtract known technical variation rather than encode it.

4. decode() returns log1p-scale expression for all genes, including unmeasured
   ones. Predicting a gene the source dataset never measured is meaningful --
   that is imputation, and it is what makes transcriptome-wide prediction
   possible -- it simply is not scored during training.
"""

from __future__ import annotations

from typing import Dict, Iterable, Optional

import torch
from torch import nn

CHECKPOINT_FORMAT = "stage1_masked_vae_v1"


class GradientReversal(torch.autograd.Function):
    """Identity forward; sign-flipped, scaled gradient backward.

    The mechanism behind adversarial batch invariance. A study classifier sits
    on the latent and is trained normally to predict which study a cell came
    from. Because its gradient reaches the encoder reversed, the encoder is
    pushed in the opposite direction -- to make study identity *unpredictable*
    from the latent.

    The point, versus conditioning the decoder on a batch covariate: study
    labels are used ONLY during training. At inference the encoder takes any
    cell with no label at all, so a study never seen in training is not a
    problem. That is the property `source_study` conditioning cannot provide,
    and measurements on this data showed conditioning only reduces study
    clustering from 14.3x to 10.4x chance anyway -- a partial fix bought at the
    cost of new-dataset transfer.
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, lambd: float) -> torch.Tensor:
        ctx.lambd = float(lambd)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):  # type: ignore[override]
        return -ctx.lambd * grad_output, None


class MaskedGaussianVAE(nn.Module):
    """Conditional Gaussian VAE whose reconstruction is masked per cell."""

    def __init__(
        self,
        n_genes: int,
        latent_dim: int = 128,
        hidden_dim: int = 512,
        n_layers: int = 2,
        dropout: float = 0.1,
        n_batches: int = 1,
        batch_embed_dim: int = 16,
        n_adversarial_classes: int = 0,
        adversarial_hidden: int = 128,
    ):
        super().__init__()
        if n_layers < 1:
            raise ValueError("n_layers must be at least 1")
        self.n_genes = int(n_genes)
        self.latent_dim = int(latent_dim)
        self.n_batches = int(n_batches)
        # A single-batch model carries no embedding at all, so an unconditioned
        # run is exactly a plain VAE rather than one with a constant input.
        self.batch_embed_dim = int(batch_embed_dim) if self.n_batches > 1 else 0
        if self.batch_embed_dim:
            self.batch_embedding = nn.Embedding(self.n_batches, self.batch_embed_dim)

        def mlp(in_dim: int, out_dim: int) -> nn.Sequential:
            layers: list[nn.Module] = [nn.LayerNorm(in_dim)]
            current = in_dim
            for _ in range(n_layers):
                layers += [nn.Linear(current, hidden_dim), nn.GELU(), nn.Dropout(dropout)]
                current = hidden_dim
            layers.append(nn.Linear(current, out_dim))
            return nn.Sequential(*layers)

        self.encoder_body = mlp(self.n_genes + self.batch_embed_dim, hidden_dim)
        self.encoder_mu = nn.Linear(hidden_dim, self.latent_dim)
        self.encoder_logvar = nn.Linear(hidden_dim, self.latent_dim)
        self.decoder = mlp(self.latent_dim + self.batch_embed_dim, self.n_genes)

        # Adversarial study classifier. Deliberately separate from the batch
        # embedding above: conditioning and adversarial invariance are two
        # different strategies, and combining both would let the decoder use a
        # batch label the encoder is simultaneously being trained to forget.
        self.n_adversarial_classes = int(n_adversarial_classes)
        if self.n_adversarial_classes > 1:
            if self.batch_embed_dim:
                raise ValueError(
                    "Adversarial invariance and batch conditioning are alternatives, not "
                    "complements: conditioning hands the decoder a label the encoder is "
                    "being trained to discard, and inference would still require that "
                    "label. Use n_batches=1 with adversarial training."
                )
            self.adversary = nn.Sequential(
                nn.Linear(self.latent_dim, adversarial_hidden),
                nn.LayerNorm(adversarial_hidden),
                nn.GELU(),
                nn.Linear(adversarial_hidden, self.n_adversarial_classes),
            )

    def adversarial_logits(self, z: torch.Tensor, lambd: float) -> torch.Tensor:
        """Study logits from a gradient-reversed latent."""
        if self.n_adversarial_classes <= 1:
            raise ValueError("model has no adversarial head")
        return self.adversary(GradientReversal.apply(z, lambd))

    def _with_batch(self, x: torch.Tensor, batch_index: Optional[torch.Tensor]) -> torch.Tensor:
        if not self.batch_embed_dim:
            return x
        if batch_index is None:
            raise ValueError("this model was trained with batch conditioning; batch_index required")
        emb = self.batch_embedding(batch_index.view(-1))
        return torch.cat([x, emb], dim=1)

    def posterior(
        self, x: torch.Tensor, batch_index: Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder_body(self._with_batch(x, batch_index))
        return self.encoder_mu(h), self.encoder_logvar(h).clamp(-12.0, 12.0)

    def encode(
        self, x: torch.Tensor, batch_index: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Deterministic posterior mean, matching the Stage-2 encode contract."""
        mu, _ = self.posterior(x, batch_index)
        return mu

    def decode(
        self, z: torch.Tensor, batch_index: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        return self.decoder(self._with_batch(z, batch_index))

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return mu
        return mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)

    def forward(
        self, x: torch.Tensor, batch_index: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        mu, logvar = self.posterior(x, batch_index)
        z = self.reparameterize(mu, logvar)
        return {"reconstruction": self.decode(z, batch_index), "mu": mu, "logvar": logvar, "z": z}


def masked_vae_loss(
    outputs: Dict[str, torch.Tensor],
    x: torch.Tensor,
    mask: Optional[torch.Tensor],
    weight: Optional[torch.Tensor] = None,
    kl_weight: float = 1e-3,
) -> Dict[str, torch.Tensor]:
    """
    Masked reconstruction + KL.

    mask: [B, G] float/bool, 1 where the cell's dataset measured that gene.
          None scores every gene, i.e. the ordinary unmasked VAE.
    weight: optional [B] per-cell weight (MOSCOT probability).

    The squared error is averaged over each cell's MEASURED genes before being
    averaged across cells, so a cell measured on a 40k panel and one measured on
    an 11k panel contribute equally. Normalizing by the gene count instead would
    let wide-panel datasets dominate purely by virtue of panel width.
    """
    reconstruction = outputs["reconstruction"]
    squared_error = (reconstruction - x).pow(2)

    if mask is None:
        per_cell = squared_error.mean(dim=1)
        measured = torch.full_like(per_cell, float(x.shape[1]))
    else:
        mask = mask.to(dtype=squared_error.dtype)
        measured = mask.sum(dim=1)
        # A cell with an empty mask contributes no reconstruction signal; guard
        # the division rather than emitting NaN into the gradient.
        per_cell = (squared_error * mask).sum(dim=1) / measured.clamp(min=1.0)

    kl_per_cell = -0.5 * (
        1.0 + outputs["logvar"] - outputs["mu"].pow(2) - outputs["logvar"].exp()
    ).sum(dim=1)

    if weight is None:
        recon = per_cell.mean()
        kl = kl_per_cell.mean()
    else:
        w = weight.to(dtype=per_cell.dtype).clamp(min=0.0)
        denom = w.sum().clamp(min=1e-8)
        recon = (per_cell * w).sum() / denom
        kl = (kl_per_cell * w).sum() / denom

    return {
        "loss": recon + kl_weight * kl,
        "reconstruction": recon,
        "kl": kl,
        "mean_measured_genes": measured.mean(),
    }


def adversarial_loss(
    model: "MaskedGaussianVAE",
    z: torch.Tensor,
    study_index: torch.Tensor,
    lambd: float,
) -> Dict[str, torch.Tensor]:
    """
    Cross-entropy of the gradient-reversed study classifier, plus its accuracy.

    Accuracy is the number to watch, and it is more interpretable than the
    loss: it should FALL toward chance (1 / n_studies) as training proceeds.
    An accuracy that stays high means the encoder is still writing study
    identity into the latent and the adversary is not winning; one that
    collapses to chance immediately usually means the latent has been degraded
    rather than made invariant -- check the role metric before believing it.
    """
    logits = model.adversarial_logits(z, lambd)
    loss = torch.nn.functional.cross_entropy(logits, study_index)
    with torch.no_grad():
        accuracy = (logits.argmax(dim=1) == study_index).float().mean()
    return {"adversarial": loss, "adversarial_accuracy": accuracy}


class MaskedVAERepresentation(nn.Module):
    """
    Frozen Stage-1 wrapper exposing the same contract as
    ScVIExpressionAutoencoder: encode(x, batch_categories) and
    decode(z, batch_categories), so Stage 2 can consume either interchangeably.
    """

    def __init__(
        self,
        module_init_kwargs: Dict[str, object],
        module_state_dict: Dict[str, torch.Tensor],
        batch_category_to_index: Optional[Dict[str, int]] = None,
    ):
        super().__init__()
        self.module_init_kwargs = dict(module_init_kwargs)
        self.vae = MaskedGaussianVAE(**self.module_init_kwargs)  # type: ignore[arg-type]
        self.vae.load_state_dict(module_state_dict, strict=True)
        self.n_latent = self.vae.latent_dim
        self.batch_category_to_index = dict(batch_category_to_index or {})
        # Mirrors the scVI wrapper: log1p in, log1p out, no likelihood branch.
        self.gene_likelihood = "masked_gaussian"

    def _batch_index(
        self, batch_categories: Optional[Iterable[str]], batch_size: int, device: torch.device
    ) -> Optional[torch.Tensor]:
        if not self.vae.batch_embed_dim:
            return None
        if batch_categories is None:
            raise ValueError(
                "This checkpoint was trained with batch conditioning "
                f"({len(self.batch_category_to_index)} categories); batch_categories required."
            )
        categories = list(batch_categories)
        if len(categories) != batch_size:
            raise ValueError(
                f"batch_categories has {len(categories)} entries but batch_size is {batch_size}"
            )
        try:
            idx = [self.batch_category_to_index[str(c)] for c in categories]
        except KeyError as exc:
            raise ValueError(
                f"Unseen batch category {exc.args[0]!r}; known: "
                f"{sorted(self.batch_category_to_index)[:8]}"
            ) from exc
        return torch.tensor(idx, dtype=torch.long, device=device)

    def encode(
        self, x: torch.Tensor, batch_categories: Optional[Iterable[str]] = None
    ) -> torch.Tensor:
        return self.vae.encode(x, self._batch_index(batch_categories, x.shape[0], x.device))

    def decode(
        self, z: torch.Tensor, batch_categories: Optional[Iterable[str]] = None
    ) -> torch.Tensor:
        return self.vae.decode(z, self._batch_index(batch_categories, z.shape[0], z.device))

    def forward(
        self, x: torch.Tensor, batch_categories: Optional[Iterable[str]] = None
    ) -> torch.Tensor:
        return self.decode(self.encode(x, batch_categories), batch_categories)

    @classmethod
    def from_checkpoint(cls, path) -> "MaskedVAERepresentation":
        try:
            payload = torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            payload = torch.load(path, map_location="cpu")
        if payload.get("format") != CHECKPOINT_FORMAT:
            raise ValueError(f"Unsupported masked-VAE checkpoint format: {payload.get('format')!r}")
        model = cls(
            module_init_kwargs=payload["module_init_kwargs"],
            module_state_dict=payload["module_state_dict"],
            batch_category_to_index=payload.get("batch_category_to_index"),
        )
        model.checkpoint_metadata = {
            "n_genes": payload["n_genes"],
            "gene_ids": payload["gene_ids"],
            "gene_symbols": payload.get("gene_symbols"),
            "batch_key": payload.get("batch_key"),
            "expression_transform": payload.get("expression_transform", "log1p_10k"),
            "min_cells_detected": payload.get("min_cells_detected", 0),
            "keep_technologies": payload.get("keep_technologies"),
            "masked": payload.get("masked", True),
        }
        return model
