"""학습 손실 함수 모음."""

import torch
import torch.nn.functional as F


def weighted_ce_loss(logits: torch.Tensor, labels: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """토큰별 가중치를 적용한 cross entropy 손실.

    인자:
        logits (torch.Tensor): (B, T, V) 로짓 텐서.
        labels (torch.Tensor): (B, T) 라벨 텐서.
        weights (torch.Tensor): (B, T) 가중치 텐서.

    반환:
        torch.Tensor: 스칼라 손실.
    """
    vocab = logits.shape[-1]
    loss = F.cross_entropy(logits.reshape(-1, vocab), labels.reshape(-1), reduction="none")
    loss = loss.reshape(labels.shape)
    weighted = loss * weights
    denom = weights.sum().clamp(min=1.0)
    return weighted.sum() / denom


def diagnosis_loss(
    diag_logits: torch.Tensor,
    diag_labels: torch.Tensor,
    loss_type: str = "bce",
) -> torch.Tensor:
    """진단 손실(BCE 또는 CE).

    인자:
        diag_logits (torch.Tensor): (B, C) 또는 (B, C, K) 진단 로짓.
        diag_labels (torch.Tensor): (B, C) 라벨(BCE는 float, CE는 int).
        loss_type (str): "bce" | "ce3" | "ce4".

    반환:
        torch.Tensor: 스칼라 손실.
    """
    if loss_type == "bce":
        return F.binary_cross_entropy_with_logits(diag_logits, diag_labels)
    if diag_logits.dim() != 3:
        raise ValueError("CE loss expects diag_logits with shape (B, C, K).")
    b, c, k = diag_logits.shape
    logits = diag_logits.view(b * c, k)
    labels = diag_labels.view(-1).long()
    return F.cross_entropy(logits, labels)
