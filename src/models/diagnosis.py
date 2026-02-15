"""진단 라벨 예측을 위한 선형 헤드."""

from __future__ import annotations

import torch
import torch.nn as nn


class DiagnosisHead(nn.Module):
    """글로벌 피처를 다중 라벨 로짓으로 투영한다."""

    def __init__(self, d_model: int, num_labels: int = 14, num_classes: int = 1) -> None:
        """진단 헤드를 초기화한다.

        인자:
            d_model (int): 입력 피처 차원.
            num_labels (int): 라벨 개수.
            num_classes (int): 각 라벨의 클래스 개수(BCE=1, CE=3/4).
        """
        super().__init__()
        self.num_labels = num_labels
        self.num_classes = num_classes
        self.proj = nn.Linear(d_model, num_labels * num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B,D) 입력을 (B,C) 또는 (B,C,K) 로짓으로 변환한다.

        인자:
            x (torch.Tensor): 입력 피처 텐서.

        반환:
            torch.Tensor: 진단 로짓 텐서.
        """
        logits = self.proj(x)
        if self.num_classes == 1:
            return logits
        b = logits.shape[0]
        return logits.view(b, self.num_labels, self.num_classes)
