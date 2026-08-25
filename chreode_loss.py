#!/usr/bin/env python3
"""
Paper-style Chreode loss components for first-round model testing.

The paper combines:

    L = lambda_mmd L_mmd + lambda_w2 L_w2 + lambda_drift L_drift + lambda_down L_down

The original setting is population-level unpaired snapshots. This project has
MOSCOT paired source/target rows, so L_drift is implemented as a paired
displacement-consistency term:

    (z_pred - z_source) should match (z_target - z_source)

MMD and Sinkhorn W2 are still population matching terms between predicted and
target latent batches. L_down is active only when the model returns a
potential_flow component, which currently means the Waddington prototype.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Sequence

import torch
import torch.nn.functional as F


DEFAULT_MMD_BANDWIDTHS = (0.001, 0.01, 0.1, 1.0, 10.0, 100.0)


@dataclass
class ChreodeLossConfig:
    lambda_mmd: float = 1.0
    lambda_w2: float = 1.0
    lambda_drift: float = 1.0
    lambda_down: float = 0.1
    sinkhorn_epsilon: float = 0.1
    sinkhorn_iters: int = 100
    mmd_bandwidths: Sequence[float] = DEFAULT_MMD_BANDWIDTHS


def _normalize_weights(weight: torch.Tensor) -> torch.Tensor:
    weight = torch.nan_to_num(weight.float(), nan=0.0, posinf=0.0, neginf=0.0)
    weight = torch.clamp(weight, min=0.0)
    total = weight.sum()
    if total <= 1e-8:
        return torch.full_like(weight, 1.0 / max(weight.numel(), 1))
    return weight / total


def _weighted_mean(values: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    w = _normalize_weights(weight).view(-1, *([1] * (values.ndim - 1)))
    return (values * w).sum(dim=0)


def weighted_latent_mse(pred: torch.Tensor, target: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    per_pair = (pred - target).pow(2).mean(dim=1)
    return (per_pair * _normalize_weights(weight)).sum()


def rbf_mmd(
    pred: torch.Tensor,
    target: torch.Tensor,
    weight: torch.Tensor,
    bandwidths: Iterable[float] = DEFAULT_MMD_BANDWIDTHS,
) -> torch.Tensor:
    """Weighted multi-bandwidth RBF MMD."""
    bandwidths = tuple(float(bw) for bw in bandwidths)
    wx = _normalize_weights(weight)
    wy = torch.full((target.shape[0],), 1.0 / target.shape[0], device=target.device, dtype=target.dtype)

    d_xx = torch.cdist(pred, pred).pow(2)
    d_yy = torch.cdist(target, target).pow(2)
    d_xy = torch.cdist(pred, target).pow(2)

    loss = pred.new_zeros(())
    for bw in bandwidths:
        gamma = 1.0 / (2.0 * float(bw))
        k_xx = torch.exp(-gamma * d_xx)
        k_yy = torch.exp(-gamma * d_yy)
        k_xy = torch.exp(-gamma * d_xy)
        loss = loss + (wx[:, None] * wx[None, :] * k_xx).sum()
        loss = loss + (wy[:, None] * wy[None, :] * k_yy).sum()
        loss = loss - 2.0 * (wx[:, None] * wy[None, :] * k_xy).sum()
    return loss / max(len(bandwidths), 1)


def sinkhorn_w2(
    pred: torch.Tensor,
    target: torch.Tensor,
    weight: torch.Tensor,
    epsilon: float = 0.1,
    n_iters: int = 100,
) -> torch.Tensor:
    """Entropic Sinkhorn transport cost with squared Euclidean ground cost."""
    a = _normalize_weights(weight)
    b = torch.full((target.shape[0],), 1.0 / target.shape[0], device=target.device, dtype=target.dtype)
    cost = torch.cdist(pred, target).pow(2)

    log_a = torch.log(a + 1e-8)
    log_b = torch.log(b + 1e-8)
    log_k = -cost / epsilon
    u = torch.zeros_like(a)
    v = torch.zeros_like(b)

    for _ in range(n_iters):
        u = log_a - torch.logsumexp(log_k + v.unsqueeze(0), dim=1)
        v = log_b - torch.logsumexp(log_k.transpose(0, 1) + u.unsqueeze(0), dim=1)

    log_plan = log_k + u.unsqueeze(1) + v.unsqueeze(0)
    plan = torch.exp(log_plan)
    return (plan * cost).sum()


def paired_drifting_loss(
    z_source: torch.Tensor,
    z_pred: torch.Tensor,
    z_target: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    """Paired adaptation of the paper's drifting-field term for MOSCOT pairs."""
    pred_step = z_pred - z_source
    target_step = z_target - z_source
    return weighted_latent_mse(pred_step, target_step, weight)


def downhill_loss(outputs: Dict[str, torch.Tensor], weight: torch.Tensor) -> torch.Tensor:
    """
    Penalize deterministic motion against the downhill potential-flow direction.

    Active for the Waddington prototype when potential_flow is available. Other
    model levels return zero for this term.
    """
    potential_flow = outputs.get("potential_flow")
    residual = outputs.get("residual")
    if potential_flow is None or residual is None:
        ref = outputs["z_pred"]
        return ref.new_zeros(())

    uphill_score = -(residual * potential_flow).sum(dim=1)
    return (F.relu(uphill_score) * _normalize_weights(weight)).sum()


def chreode_population_loss(
    outputs: Dict[str, torch.Tensor],
    z_target: torch.Tensor,
    weight: torch.Tensor,
    config: ChreodeLossConfig,
) -> Dict[str, torch.Tensor]:
    z_source = outputs["z_source"]
    z_pred = outputs["z_pred"]

    loss_mmd = rbf_mmd(z_pred, z_target, weight, config.mmd_bandwidths)
    loss_w2 = sinkhorn_w2(z_pred, z_target, weight, config.sinkhorn_epsilon, config.sinkhorn_iters)
    loss_drift = paired_drifting_loss(z_source, z_pred, z_target, weight)
    loss_down = downhill_loss(outputs, weight)

    total = (
        config.lambda_mmd * loss_mmd
        + config.lambda_w2 * loss_w2
        + config.lambda_drift * loss_drift
        + config.lambda_down * loss_down
    )
    return {
        "loss": total,
        "loss_mmd": loss_mmd,
        "loss_w2": loss_w2,
        "loss_drift": loss_drift,
        "loss_down": loss_down,
    }
