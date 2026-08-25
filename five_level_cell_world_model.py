#!/usr/bin/env python3
"""
Six-level one-step temporal cell world model.

The model trains directly on the paired h5ad dataloader contract:

    batch["x"]      source / baseline expression, [B, G]
    batch["y"]      target / posttreatment expression, [B, G]
    batch["delta"]  elapsed time, [B], defaults to one step
    batch["weight"] pair-level MOSCOT probability weight, [B]

This file contains first-pass prototypes for the transition levels. The
real representation should be a pretrained scVI latent space. The small
expression autoencoder below is only a temporary adapter so the current paired
h5ad dataloader can be used immediately for smoke tests.
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from typing import Dict, Optional, Sequence

import math

import torch
from torch import nn
import torch.nn.functional as F


def weighted_mse(pred: torch.Tensor, target: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Pair-weighted MSE where one weight applies to one full cell vector."""
    per_pair = (pred - target).pow(2).mean(dim=1)
    return (per_pair * weight).sum() / (weight.sum() + 1e-8)


def sdpa_math_kernel_context():
    """
    Force math scaled-dot-product attention on CUDA.

    The Waddington prototype takes gradients through attention to compute
    -grad_z U. Some efficient attention kernels do not implement that derivative.
    """
    if not torch.cuda.is_available():
        return nullcontext()

    attention_mod = getattr(torch.nn, "attention", None)
    if attention_mod is not None and hasattr(attention_mod, "sdpa_kernel"):
        backend = getattr(attention_mod.SDPBackend, "MATH", None)
        if backend is not None:
            return attention_mod.sdpa_kernel(backend)

    return torch.backends.cuda.sdp_kernel(
        enable_flash=False,
        enable_math=True,
        enable_mem_efficient=False,
    )


class FourierTimeEmbedding(nn.Module):
    """Continuous elapsed-time embedding for scalar delta."""

    def __init__(self, dim: int, max_period: float = 10_000.0):
        super().__init__()
        self.dim = dim
        self.max_period = max_period
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, delta: torch.Tensor) -> torch.Tensor:
        delta = delta.float().view(-1, 1)
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(self.max_period)
            * torch.arange(half, device=delta.device, dtype=torch.float32)
            / max(half - 1, 1)
        )
        args = delta * freqs.view(1, -1)
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=1)
        if emb.shape[1] < self.dim:
            emb = F.pad(emb, (0, self.dim - emb.shape[1]))
        return self.mlp(emb)


class ExpressionAutoencoder(nn.Module):
    """Legacy deterministic adapter retained only for old-checkpoint compatibility."""

    def __init__(self, n_genes: int, latent_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.LayerNorm(n_genes),
            nn.Linear(n_genes, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_genes),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)


class GaussianExpressionVAE(nn.Module):
    """Gaussian VAE for normalize_total(1e4)+log1p expression (paper Stage 1)."""

    def __init__(
        self,
        n_genes: int,
        latent_dim: int,
        hidden_dim: int,
        dropout: float,
        depth: int = 3,
    ):
        super().__init__()
        if depth < 1:
            raise ValueError("VAE depth must be positive")

        encoder_layers = [nn.LayerNorm(n_genes)]
        in_dim = n_genes
        for _ in range(depth):
            encoder_layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
            in_dim = hidden_dim
        self.encoder_body = nn.Sequential(*encoder_layers)
        self.encoder_mu = nn.Linear(hidden_dim, latent_dim)
        self.encoder_logvar = nn.Linear(hidden_dim, latent_dim)

        decoder_layers = [nn.LayerNorm(latent_dim)]
        in_dim = latent_dim
        for _ in range(depth):
            decoder_layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
            in_dim = hidden_dim
        decoder_layers.append(nn.Linear(hidden_dim, n_genes))
        self.decoder = nn.Sequential(*decoder_layers)

    def posterior(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder_body(x)
        return self.encoder_mu(h), self.encoder_logvar(h).clamp(-12.0, 12.0)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Return the deterministic posterior mean used by Stage 2."""
        mu, _ = self.posterior(x)
        return mu

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return mu
        return mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        mu, logvar = self.posterior(x)
        z = self.reparameterize(mu, logvar)
        return {
            "reconstruction": self.decode(z),
            "mu": mu,
            "logvar": logvar,
            "z": z,
        }


class LatentTokenizer(nn.Module):
    """Project one latent vector into K Transformer tokens."""

    def __init__(self, latent_dim: int, model_dim: int, num_tokens: int):
        super().__init__()
        self.num_tokens = num_tokens
        self.model_dim = model_dim
        self.proj = nn.Linear(latent_dim, num_tokens * model_dim)
        self.pos = nn.Parameter(torch.zeros(1, num_tokens, model_dim))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        tokens = self.proj(z).view(z.shape[0], self.num_tokens, self.model_dim)
        return tokens + self.pos


class IdentityResidual(nn.Module):
    """Level 0: non-learned baseline. Predicts no change (z_pred = z_source)."""

    def forward(self, z: torch.Tensor, delta: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {"z_pred": z, "residual": torch.zeros_like(z)}


class PlainTransformerResidual(nn.Module):
    """Level 1: generic Transformer residual model."""

    def __init__(self, latent_dim: int, model_dim: int, depth: int, heads: int, num_tokens: int, dropout: float):
        super().__init__()
        self.time_embed = FourierTimeEmbedding(model_dim)
        self.tokenizer = LatentTokenizer(latent_dim, model_dim, num_tokens)
        layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=heads,
            dim_feedforward=model_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.blocks = nn.TransformerEncoder(layer, num_layers=depth)
        self.head = nn.Sequential(nn.LayerNorm(model_dim), nn.Linear(model_dim, latent_dim))

    def forward(self, z: torch.Tensor, delta: torch.Tensor) -> Dict[str, torch.Tensor]:
        time_token = self.time_embed(delta).unsqueeze(1)
        tokens = torch.cat([time_token, self.tokenizer(z)], dim=1)
        h = self.blocks(tokens).mean(dim=1)
        residual = self.head(h)
        return {"z_pred": z + residual, "residual": residual}


class AdaLNZeroBlock(nn.Module):
    """DiT-style block with adaptive LayerNorm modulation initialized near zero."""

    def __init__(self, model_dim: int, heads: int, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(model_dim, elementwise_affine=False)
        self.attn = nn.MultiheadAttention(model_dim, heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(model_dim, elementwise_affine=False)
        self.mlp = nn.Sequential(
            nn.Linear(model_dim, model_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(model_dim * 4, model_dim),
        )
        self.mod = nn.Sequential(nn.SiLU(), nn.Linear(model_dim, model_dim * 6))
        nn.init.zeros_(self.mod[-1].weight)
        nn.init.zeros_(self.mod[-1].bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        s1, b1, g1, s2, b2, g2 = self.mod(cond).chunk(6, dim=1)
        h = self.norm1(x) * (1 + s1.unsqueeze(1)) + b1.unsqueeze(1)
        with sdpa_math_kernel_context():
            attn, _ = self.attn(h, h, h, need_weights=False)
        x = x + g1.unsqueeze(1) * attn
        h = self.norm2(x) * (1 + s2.unsqueeze(1)) + b2.unsqueeze(1)
        x = x + g2.unsqueeze(1) * self.mlp(h)
        return x


class DiTBackbone(nn.Module):
    """Tokenized latent DiT backbone shared by levels 2-5."""

    def __init__(self, latent_dim: int, model_dim: int, depth: int, heads: int, num_tokens: int, dropout: float):
        super().__init__()
        self.time_embed = FourierTimeEmbedding(model_dim)
        self.tokenizer = LatentTokenizer(latent_dim, model_dim, num_tokens)
        self.blocks = nn.ModuleList([AdaLNZeroBlock(model_dim, heads, dropout) for _ in range(depth)])
        self.norm = nn.LayerNorm(model_dim)

    def forward(self, z: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
        cond = self.time_embed(delta)
        x = self.tokenizer(z)
        for block in self.blocks:
            x = block(x, cond)
        return self.norm(x).mean(dim=1)


class DiTTokenBackbone(nn.Module):
    """Tokenized DiT backbone that keeps per-token features for richer variants."""

    def __init__(self, latent_dim: int, model_dim: int, depth: int, heads: int, num_tokens: int, dropout: float):
        super().__init__()
        self.time_embed = FourierTimeEmbedding(model_dim)
        self.tokenizer = LatentTokenizer(latent_dim, model_dim, num_tokens)
        self.blocks = nn.ModuleList([AdaLNZeroBlock(model_dim, heads, dropout) for _ in range(depth)])
        self.norm = nn.LayerNorm(model_dim)

    def forward_tokens(self, z: torch.Tensor, delta: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        cond = self.time_embed(delta)
        x = self.tokenizer(z)
        for block in self.blocks:
            x = block(x, cond)
        return self.norm(x), cond

    def forward(self, z: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
        tokens, _ = self.forward_tokens(z, delta)
        return tokens.mean(dim=1)


class DiTResidual(nn.Module):
    """Level 2: DiT residual model."""

    def __init__(self, latent_dim: int, model_dim: int, depth: int, heads: int, num_tokens: int, dropout: float):
        super().__init__()
        self.backbone = DiTBackbone(latent_dim, model_dim, depth, heads, num_tokens, dropout)
        self.head = nn.Linear(model_dim, latent_dim)

    def forward(self, z: torch.Tensor, delta: torch.Tensor) -> Dict[str, torch.Tensor]:
        residual = self.head(self.backbone(z, delta))
        return {"z_pred": z + residual, "residual": residual}


class TimeGate(nn.Module):
    """alpha(delta) = 1 - exp(-delta / tau), with learnable positive tau."""

    def __init__(self, init_tau: float = 1.0):
        super().__init__()
        self.raw_tau = nn.Parameter(torch.tensor(float(init_tau)).log())

    def forward(self, delta: torch.Tensor) -> torch.Tensor:
        tau = F.softplus(self.raw_tau) + 1e-6
        clean_delta = torch.clamp(delta.float(), min=0.0)
        return (1.0 - torch.exp(-clean_delta / tau)).view(-1, 1)


class GatedDiTResidual(nn.Module):
    """Level 3: DiT residual with explicit elapsed-time gate."""

    def __init__(self, latent_dim: int, model_dim: int, depth: int, heads: int, num_tokens: int, dropout: float):
        super().__init__()
        self.backbone = DiTBackbone(latent_dim, model_dim, depth, heads, num_tokens, dropout)
        self.head = nn.Linear(model_dim, latent_dim)
        self.gate = TimeGate()

    def forward(self, z: torch.Tensor, delta: torch.Tensor) -> Dict[str, torch.Tensor]:
        residual = self.head(self.backbone(z, delta))
        alpha = self.gate(delta)
        return {"z_pred": z + alpha * residual, "residual": residual, "alpha": alpha}


class WaddingtonResidualDiT(nn.Module):
    """Level 4: Chreode-style potential + rotational + stochastic residual DiT."""

    def __init__(self, latent_dim: int, model_dim: int, depth: int, heads: int, num_tokens: int, dropout: float):
        super().__init__()
        self.latent_dim = latent_dim
        self.backbone = DiTBackbone(latent_dim, model_dim, depth, heads, num_tokens, dropout)
        self.potential = nn.Linear(model_dim, 1)
        self.rotation = nn.Linear(model_dim, latent_dim * latent_dim)
        self.log_sigma = nn.Linear(model_dim, latent_dim)
        self.gate = TimeGate()

    def forward(self, z: torch.Tensor, delta: torch.Tensor) -> Dict[str, torch.Tensor]:
        with torch.enable_grad():
            z_in = z.requires_grad_(True)
            h = self.backbone(z_in, delta)
            potential = self.potential(h).sum()
            grad = torch.autograd.grad(potential, z_in, create_graph=self.training, retain_graph=True)[0]
        potential_flow = -grad

        raw_matrix = self.rotation(h).view(-1, self.latent_dim, self.latent_dim)
        skew = raw_matrix - raw_matrix.transpose(1, 2)
        rotational_flow = torch.bmm(skew, z_in.unsqueeze(-1)).squeeze(-1) / math.sqrt(self.latent_dim)

        sigma = F.softplus(self.log_sigma(h)) + 1e-6
        eps = torch.randn_like(sigma) if self.training else torch.zeros_like(sigma)
        stochastic_flow = sigma * eps

        residual = potential_flow + rotational_flow + stochastic_flow
        alpha = self.gate(delta)
        return {
            "z_pred": z + alpha * residual,
            "residual": residual,
            "alpha": alpha,
            "potential_flow": potential_flow,
            "rotational_flow": rotational_flow,
            "sigma": sigma,
        }


class AdvancedDiTResidual(nn.Module):
    """Level 5 prototype: tokenized AdaLN-Zero DiT with a stochastic residual head."""

    def __init__(self, latent_dim: int, model_dim: int, depth: int, heads: int, num_tokens: int, dropout: float):
        super().__init__()
        self.backbone = DiTBackbone(latent_dim, model_dim, depth, heads, num_tokens, dropout)
        self.mean = nn.Linear(model_dim, latent_dim)
        self.log_sigma = nn.Linear(model_dim, latent_dim)
        self.gate = TimeGate()

    def forward(self, z: torch.Tensor, delta: torch.Tensor) -> Dict[str, torch.Tensor]:
        h = self.backbone(z, delta)
        mean = self.mean(h)
        sigma = F.softplus(self.log_sigma(h)) + 1e-6
        eps = torch.randn_like(sigma) if self.training else torch.zeros_like(sigma)
        residual = mean + sigma * eps
        alpha = self.gate(delta)
        return {
            "z_pred": z + alpha * residual,
            "residual": residual,
            "alpha": alpha,
            "residual_mean": mean,
            "sigma": sigma,
        }


class TokenizedLatentDiTResidual(nn.Module):
    """Level 5B: deterministic tokenized latent AdaLN-Zero DiT."""

    def __init__(self, latent_dim: int, model_dim: int, depth: int, heads: int, num_tokens: int, dropout: float):
        super().__init__()
        self.backbone = DiTTokenBackbone(latent_dim, model_dim, depth, heads, num_tokens, dropout)
        self.head = nn.Linear(model_dim, latent_dim)
        self.gate = TimeGate()

    def forward(self, z: torch.Tensor, delta: torch.Tensor) -> Dict[str, torch.Tensor]:
        h = self.backbone(z, delta)
        residual = self.head(h)
        alpha = self.gate(delta)
        return {"z_pred": z + alpha * residual, "residual": residual, "alpha": alpha}


class CrossAttentionDiTResidual(nn.Module):
    """
    Level 5D: tokenized DiT with a cross-attention context path.

    The current dataloader does not yet provide intervention/action tokens, so
    this screening prototype uses a learned null context plus elapsed time.
    """

    def __init__(self, latent_dim: int, model_dim: int, depth: int, heads: int, num_tokens: int, dropout: float):
        super().__init__()
        self.backbone = DiTTokenBackbone(latent_dim, model_dim, depth, heads, num_tokens, dropout)
        self.context_token = nn.Parameter(torch.zeros(1, 1, model_dim))
        self.context_norm = nn.LayerNorm(model_dim)
        self.query_norm = nn.LayerNorm(model_dim)
        self.cross_attn = nn.MultiheadAttention(model_dim, heads, dropout=dropout, batch_first=True)
        self.out_norm = nn.LayerNorm(model_dim)
        self.head = nn.Linear(model_dim, latent_dim)
        self.gate = TimeGate()

    def forward(self, z: torch.Tensor, delta: torch.Tensor) -> Dict[str, torch.Tensor]:
        tokens, cond = self.backbone.forward_tokens(z, delta)
        context = self.context_token.expand(z.shape[0], -1, -1) + cond.unsqueeze(1)
        with sdpa_math_kernel_context():
            attended, _ = self.cross_attn(
                self.query_norm(tokens),
                self.context_norm(context),
                self.context_norm(context),
                need_weights=False,
            )
        h = self.out_norm(tokens + attended).mean(dim=1)
        residual = self.head(h)
        alpha = self.gate(delta)
        return {"z_pred": z + alpha * residual, "residual": residual, "alpha": alpha}


class HierarchicalUNetDiTResidual(nn.Module):
    """Level 5E: small hierarchical/U-Net-inspired token DiT."""

    def __init__(self, latent_dim: int, model_dim: int, depth: int, heads: int, num_tokens: int, dropout: float):
        super().__init__()
        pre_depth = max(1, depth // 2)
        post_depth = max(1, depth - pre_depth)
        self.time_embed = FourierTimeEmbedding(model_dim)
        self.tokenizer = LatentTokenizer(latent_dim, model_dim, num_tokens)
        self.pre_blocks = nn.ModuleList([AdaLNZeroBlock(model_dim, heads, dropout) for _ in range(pre_depth)])
        self.bottleneck = nn.Sequential(
            nn.LayerNorm(model_dim),
            nn.Linear(model_dim, model_dim),
            nn.GELU(),
            nn.Linear(model_dim, model_dim),
        )
        self.skip_scale = nn.Parameter(torch.tensor(0.0))
        self.post_blocks = nn.ModuleList([AdaLNZeroBlock(model_dim, heads, dropout) for _ in range(post_depth)])
        self.norm = nn.LayerNorm(model_dim)
        self.head = nn.Linear(model_dim, latent_dim)
        self.gate = TimeGate()

    def forward(self, z: torch.Tensor, delta: torch.Tensor) -> Dict[str, torch.Tensor]:
        cond = self.time_embed(delta)
        x = self.tokenizer(z)
        for block in self.pre_blocks:
            x = block(x, cond)
        skip = x
        pooled = self.bottleneck(x.mean(dim=1)).unsqueeze(1)
        x = x + pooled + torch.tanh(self.skip_scale) * skip
        for block in self.post_blocks:
            x = block(x, cond)
        residual = self.head(self.norm(x).mean(dim=1))
        alpha = self.gate(delta)
        return {"z_pred": z + alpha * residual, "residual": residual, "alpha": alpha}


class ConsistencyStyleDiTResidual(nn.Module):
    """Level 5F: one-step consistency-style DiT with direct and residual heads."""

    def __init__(self, latent_dim: int, model_dim: int, depth: int, heads: int, num_tokens: int, dropout: float):
        super().__init__()
        self.backbone = DiTTokenBackbone(latent_dim, model_dim, depth, heads, num_tokens, dropout)
        self.residual_head = nn.Linear(model_dim, latent_dim)
        self.direct_head = nn.Linear(model_dim, latent_dim)
        self.mix = nn.Parameter(torch.tensor(-2.0))
        self.gate = TimeGate()

    def forward(self, z: torch.Tensor, delta: torch.Tensor) -> Dict[str, torch.Tensor]:
        h = self.backbone(z, delta)
        residual_pred = z + self.gate(delta) * self.residual_head(h)
        direct_pred = self.direct_head(h)
        mix = torch.sigmoid(self.mix)
        z_pred = (1.0 - mix) * residual_pred + mix * direct_pred
        return {
            "z_pred": z_pred,
            "residual": z_pred - z,
            "consistency_direct": direct_pred,
            "consistency_mix": mix,
        }


class RectifiedFlowInspiredDiTResidual(nn.Module):
    """Level 5G: rectified-flow-inspired one-step displacement DiT."""

    def __init__(self, latent_dim: int, model_dim: int, depth: int, heads: int, num_tokens: int, dropout: float):
        super().__init__()
        self.backbone = DiTTokenBackbone(latent_dim, model_dim, depth, heads, num_tokens, dropout)
        self.velocity = nn.Linear(model_dim, latent_dim)

    def forward(self, z: torch.Tensor, delta: torch.Tensor) -> Dict[str, torch.Tensor]:
        h = self.backbone(z, delta)
        velocity = self.velocity(h)
        step = torch.clamp(delta.float(), min=0.0).view(-1, 1)
        residual = step * velocity
        return {"z_pred": z + residual, "residual": residual, "velocity": velocity}


class CellFlowVelocityField(nn.Module):
    """
    Level 6: CellFlow-style conditional flow-matching velocity field.

    CellFlow trains a vector field on conditional probability paths between
    source and target populations. Our paired MOSCOT rows provide the OT
    coupling, and this module represents v_theta(z_t, t | delta). Perturbation
    labels/actions are intentionally omitted for the current pre/post-treatment
    temporal screen; delta is only a time-gap condition.
    """

    def __init__(self, latent_dim: int, model_dim: int, depth: int, heads: int, num_tokens: int, dropout: float):
        super().__init__()
        self.flow_time_embed = FourierTimeEmbedding(model_dim)
        self.delta_embed = FourierTimeEmbedding(model_dim)
        self.tokenizer = LatentTokenizer(latent_dim, model_dim, num_tokens)
        self.blocks = nn.ModuleList([AdaLNZeroBlock(model_dim, heads, dropout) for _ in range(depth)])
        self.norm = nn.LayerNorm(model_dim)
        self.velocity_head = nn.Linear(model_dim, latent_dim)

    def forward_velocity(self, z_t: torch.Tensor, flow_time: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
        cond = self.flow_time_embed(flow_time) + self.delta_embed(delta)
        x = self.tokenizer(z_t)
        for block in self.blocks:
            x = block(x, cond)
        return self.velocity_head(self.norm(x).mean(dim=1))

    def integrate(
        self,
        z_source: torch.Tensor,
        delta: torch.Tensor,
        n_steps: int = 8,
        method: str = "euler",
    ) -> torch.Tensor:
        """
        Integrate the learned velocity field from t=0 to t=1.

        This turns the CellFlow velocity model into a comparable endpoint
        predictor for pair/MMD/W2 validation. Euler is the default because it is
        stable, cheap, and enough for architecture/loss screening; Heun is
        available for a slightly more accurate endpoint estimate.
        """
        if n_steps <= 0:
            raise ValueError("n_steps must be positive.")
        if method not in {"euler", "heun"}:
            raise ValueError("method must be 'euler' or 'heun'.")

        z = z_source
        dt = 1.0 / float(n_steps)
        batch = z_source.shape[0]
        for step in range(n_steps):
            t0 = torch.full((batch,), step * dt, device=z_source.device, dtype=z_source.dtype)
            v0 = self.forward_velocity(z, t0, delta)
            if method == "euler":
                z = z + dt * v0
                continue

            z_euler = z + dt * v0
            t1 = torch.full((batch,), min((step + 1) * dt, 1.0), device=z_source.device, dtype=z_source.dtype)
            v1 = self.forward_velocity(z_euler, t1, delta)
            z = z + 0.5 * dt * (v0 + v1)
        return z

    def forward(self, z: torch.Tensor, delta: torch.Tensor) -> Dict[str, torch.Tensor]:
        z_pred = self.integrate(z, delta, n_steps=8, method="euler")
        flow_time = torch.zeros_like(delta, dtype=z.dtype, device=z.device)
        velocity = self.forward_velocity(z, flow_time, delta)
        return {"z_pred": z_pred, "residual": z_pred - z, "velocity": velocity}


@dataclass
class CellWorldModelConfig:
    n_genes: int
    level: int = 3
    variant: str = "default"
    latent_dim: int = 128
    expression_hidden_dim: int = 1024
    expression_depth: int = 3
    representation_type: str = "legacy_mlp"
    scvi_checkpoint: Optional[str] = None
    model_dim: int = 256
    depth: int = 4
    heads: int = 8
    num_tokens: int = 8
    dropout: float = 0.1


class FiveLevelCellWorldModel(nn.Module):
    """Expression-space wrapper around the latent world-model levels."""

    def __init__(self, config: CellWorldModelConfig):
        super().__init__()
        if config.level not in {0, 1, 2, 3, 4, 5, 6}:
            raise ValueError("level must be one of {0, 1, 2, 3, 4, 5, 6}.")
        if config.level != 5 and config.variant != "default":
            raise ValueError("--variant is only used for level 5; use --variant default for other levels.")
        self.config = config
        if config.representation_type == "legacy_mlp":
            self.autoencoder = ExpressionAutoencoder(
                n_genes=config.n_genes,
                latent_dim=config.latent_dim,
                hidden_dim=config.expression_hidden_dim,
                dropout=config.dropout,
            )
        elif config.representation_type == "gaussian_vae":
            self.autoencoder = GaussianExpressionVAE(
                n_genes=config.n_genes,
                latent_dim=config.latent_dim,
                hidden_dim=config.expression_hidden_dim,
                dropout=config.dropout,
                depth=config.expression_depth,
            )
        elif config.representation_type == "scvi":
            if not config.scvi_checkpoint:
                raise ValueError("representation_type='scvi' requires config.scvi_checkpoint")
            from scvi_stage1_representation import ScVIExpressionAutoencoder

            self.autoencoder = ScVIExpressionAutoencoder.from_checkpoint(config.scvi_checkpoint)
            if self.autoencoder.checkpoint_metadata["n_genes"] != config.n_genes:
                raise ValueError(
                    "scVI checkpoint gene count "
                    f"({self.autoencoder.checkpoint_metadata['n_genes']}) does not match "
                    f"the training data ({config.n_genes})."
                )
            if self.autoencoder.n_latent != config.latent_dim:
                raise ValueError(
                    f"scVI checkpoint latent_dim ({self.autoencoder.n_latent}) does not "
                    f"match --latent-dim ({config.latent_dim})."
                )
        else:
            raise ValueError(f"Unknown representation_type: {config.representation_type}")
        transition_args = (
            config.latent_dim,
            config.model_dim,
            config.depth,
            config.heads,
            config.num_tokens,
            config.dropout,
        )
        if config.level == 0:
            self.transition = IdentityResidual()
        elif config.level == 1:
            self.transition = PlainTransformerResidual(*transition_args)
        elif config.level == 2:
            self.transition = DiTResidual(*transition_args)
        elif config.level == 3:
            self.transition = GatedDiTResidual(*transition_args)
        elif config.level == 4:
            self.transition = WaddingtonResidualDiT(*transition_args)
        elif config.level == 5:
            variants = {
                "default": AdvancedDiTResidual,
                "stochastic_dit": AdvancedDiTResidual,
                "tokenized_dit": TokenizedLatentDiTResidual,
                "cross_attention_dit": CrossAttentionDiTResidual,
                "hierarchical_dit": HierarchicalUNetDiTResidual,
                "consistency_dit": ConsistencyStyleDiTResidual,
                "rectified_dit": RectifiedFlowInspiredDiTResidual,
            }
            if config.variant == "diffusion_denoising_dit":
                raise ValueError("diffusion_denoising_dit is intentionally excluded from current screening.")
            if config.variant not in variants:
                raise ValueError(f"unknown level-5 variant: {config.variant}")
            self.transition = variants[config.variant](*transition_args)
        else:
            self.transition = CellFlowVelocityField(*transition_args)

    @property
    def autoencoder_takes_batch(self) -> bool:
        """True when the representation conditions on a technical batch label."""
        return self.config.representation_type == "scvi"

    def forward(
        self,
        x: torch.Tensor,
        delta: Optional[torch.Tensor] = None,
        batch_categories: Optional[Sequence[str]] = None,
    ) -> Dict[str, torch.Tensor]:
        if delta is None:
            delta = torch.ones(x.shape[0], device=x.device, dtype=x.dtype)
        z_source = self.encode(x, batch_categories)
        transition_out = self.transition(z_source, delta)
        y_pred = self.decode(transition_out["z_pred"], batch_categories)
        out = {
            "y_pred": y_pred,
            "z_source": z_source,
            "z_pred": transition_out["z_pred"],
            "delta": delta,
        }
        out.update(transition_out)
        return out

    def encode(
        self, x: torch.Tensor, batch_categories: Optional[Sequence[str]] = None
    ) -> torch.Tensor:
        if self.autoencoder_takes_batch:
            return self.autoencoder.encode(x, batch_categories)
        return self.autoencoder.encode(x)

    def decode(
        self, z: torch.Tensor, batch_categories: Optional[Sequence[str]] = None
    ) -> torch.Tensor:
        # A predicted post-treatment latent has no observed batch of its own, so
        # it is decoded into the same batch as the source cell it came from.
        if self.autoencoder_takes_batch:
            return self.autoencoder.decode(z, batch_categories)
        return self.autoencoder.decode(z)

    def predict_velocity(
        self,
        z_t: torch.Tensor,
        flow_time: torch.Tensor,
        delta: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.config.level != 6:
            raise ValueError("predict_velocity is only available for level 6.")
        if delta is None:
            delta = torch.ones(z_t.shape[0], device=z_t.device, dtype=z_t.dtype)
        return self.transition.forward_velocity(z_t, flow_time, delta)

    def integrate_cellflow(
        self,
        z_source: torch.Tensor,
        delta: Optional[torch.Tensor] = None,
        n_steps: int = 8,
        method: str = "euler",
    ) -> torch.Tensor:
        if self.config.level != 6:
            raise ValueError("integrate_cellflow is only available for level 6.")
        if delta is None:
            delta = torch.ones(z_source.shape[0], device=z_source.device, dtype=z_source.dtype)
        return self.transition.integrate(z_source, delta, n_steps=n_steps, method=method)


def build_model(
    n_genes: int,
    level: int,
    variant: str = "default",
    latent_dim: int = 128,
    expression_hidden_dim: int = 1024,
    expression_depth: int = 3,
    representation_type: str = "legacy_mlp",
    scvi_checkpoint: Optional[str] = None,
    model_dim: int = 256,
    depth: int = 4,
    heads: int = 8,
    num_tokens: int = 8,
    dropout: float = 0.1,
) -> FiveLevelCellWorldModel:
    config = CellWorldModelConfig(
        n_genes=n_genes,
        level=level,
        variant=variant,
        latent_dim=latent_dim,
        expression_hidden_dim=expression_hidden_dim,
        expression_depth=expression_depth,
        representation_type=representation_type,
        scvi_checkpoint=scvi_checkpoint,
        model_dim=model_dim,
        depth=depth,
        heads=heads,
        num_tokens=num_tokens,
        dropout=dropout,
    )
    return FiveLevelCellWorldModel(config)
