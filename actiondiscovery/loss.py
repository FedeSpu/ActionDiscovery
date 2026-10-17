"""Cross entropy, truncated smoothing, and unknown-only temporal contrastive loss."""

import numpy as np
import torch
from scipy.stats import truncnorm
from torch.nn import functional as F

from .config import TrainingConfig


def temporal_contrastive(
    embeddings: torch.Tensor, targets: torch.Tensor, config: TrainingConfig
) -> torch.Tensor:
    """Temporal positives and distant negatives sampled within unknown embeddings.

    Short sequences with no valid negative candidates contribute zero. Compute
    gathered cosine similarities directly rather than a quadratic similarity matrix.
    Padding (-100) and known classes never participate.
    """
    values = []
    for features, labels in zip(embeddings.transpose(1, 2), targets):
        unknown = F.normalize(features[labels == 0], dim=-1)
        length = len(unknown)
        window = length // 2 - 5
        if window < 1:
            continue
        sigma = max(length // 4, 1)
        positive, negative, anchors = [], [], []
        for anchor in range(length):
            candidates = np.concatenate(
                (
                    np.arange(max(0, anchor - window)),
                    np.arange(min(length, anchor + window + 1), length),
                )
            )
            if not len(candidates):
                continue
            lower = 0 if anchor < length // 2 else (-length // 2) / sigma
            upper = (length - anchor) / sigma
            partner = int(truncnorm.rvs(lower, upper, loc=anchor, scale=sigma))
            positive.append(min(max(partner, 0), length - 1))
            negative.append(
                np.random.choice(
                    candidates,
                    config.negatives,
                    replace=len(candidates) < config.negatives,
                )
            )
            anchors.append(anchor)
        if anchors:
            anchor_features = unknown[anchors]
            pos = (anchor_features * unknown[positive]).sum(-1, keepdim=True)
            neg = (anchor_features[:, None] * unknown[np.asarray(negative)]).sum(-1)
            logits = torch.cat((pos, neg), dim=1) / config.temperature
            values.append(
                F.cross_entropy(
                    logits,
                    torch.zeros(len(logits), dtype=torch.long, device=logits.device),
                )
            )
    return torch.stack(values).mean() if values else embeddings.sum() * 0


def training_loss(
    logits: torch.Tensor,
    embeddings: torch.Tensor,
    targets: torch.Tensor,
    config: TrainingConfig,
) -> dict[str, torch.Tensor]:
    """Sum CE and smoothing over stages; add contrastive loss once."""
    ce, smoothing = logits.sum() * 0, logits.sum() * 0
    for stage in logits:
        if (targets != -100).any():
            ce = ce + F.cross_entropy(stage, targets, ignore_index=-100)
        if stage.shape[-1] > 1:
            delta = F.mse_loss(
                F.log_softmax(stage[:, :, 1:], dim=1),
                F.log_softmax(stage.detach()[:, :, :-1], dim=1),
                reduction="none",
            )
            smoothing = (
                smoothing
                + (
                    delta.clamp(0, config.smoothing_clip)
                    * (targets[:, 1:] != -100).unsqueeze(1)
                ).mean()
            )
    contrastive = (
        temporal_contrastive(embeddings, targets, config)
        if config.contrastive_weight
        else embeddings.sum() * 0
    )
    return {
        "loss": ce
        + config.smoothing_weight * smoothing
        + config.contrastive_weight * contrastive,
        "cross_entropy": ce,
        "smoothing": smoothing,
        "contrastive": contrastive,
    }
