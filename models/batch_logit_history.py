"""Lightweight batch-level history utilities for adaptive teacher weighting."""

import torch
import torch.nn.functional as F


def compute_batch_dissimilarities(branch_mean_logits, teacher_mean_logit, metric):
    """Compare branch-level mean logits with the previous batch teacher mean.

    Args:
        branch_mean_logits: Tensor shaped ``[num_branches, num_classes]``.
        teacher_mean_logit: Tensor shaped ``[num_classes]``.
        metric: Dissimilarity metric name.

    Returns:
        One detached distance per branch, shaped ``[num_branches]``.
    """
    current_probs = F.softmax(branch_mean_logits, dim=-1)
    teacher_probs = F.softmax(teacher_mean_logit, dim=-1).unsqueeze(0)

    if metric == 'wasserstein1':
        current_cdfs = torch.cumsum(current_probs, dim=-1)
        teacher_cdf = torch.cumsum(teacher_probs, dim=-1)
        distances = torch.sum(torch.abs(current_cdfs - teacher_cdf), dim=-1)
    elif metric == 'wasserstein2':
        current_cdfs = torch.cumsum(current_probs, dim=-1)
        teacher_cdf = torch.cumsum(teacher_probs, dim=-1)
        distances = torch.sqrt(
            torch.sum((current_cdfs - teacher_cdf) ** 2, dim=-1)
        )
    elif metric == 'euclidean':
        distances = torch.sqrt(
            torch.sum((current_probs - teacher_probs) ** 2, dim=-1)
        )
    elif metric == 'kl':
        eps = 1e-8
        current_probs = current_probs + eps
        teacher_probs = teacher_probs + eps
        distances = torch.sum(
            current_probs * torch.log(current_probs / teacher_probs), dim=-1
        )
    elif metric == 'cosine':
        cosine_similarity = F.cosine_similarity(
            current_probs, teacher_probs.expand_as(current_probs), dim=-1
        )
        distances = 1 - cosine_similarity
    else:
        raise ValueError(f"Unknown dissimilarity metric: {metric}")

    return distances.detach()
