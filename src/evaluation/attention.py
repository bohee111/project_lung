"""어텐션 기반 시각화 유틸리티."""

from __future__ import annotations

from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt


def attention_to_heatmap(attn_tokens: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
    """프리픽스 어텐션을 마스크 가중합으로 변환해 히트맵을 만든다.

    인자:
        attn_tokens (torch.Tensor): 토큰별 프리픽스 어텐션 가중치.
        masks (torch.Tensor): 좌/우 폐 및 심장 마스크.

    반환:
        torch.Tensor: 정규화된 히트맵(또는 None).
    """
    # attn_tokens: (T, 4) 또는 (T, n_visual_tokens)
    if attn_tokens is None:
        return None
    weights = attn_tokens.mean(dim=0)
    weights = weights / (weights.sum() + 1e-6)

    global_w = weights[0].item()
    local_w = weights[1:4]

    masks = masks.squeeze(0)  # 마스크 텐서 (3, H, W)
    h, w = masks.shape[1], masks.shape[2]
    heatmap = torch.ones((h, w)) * global_w
    for i in range(3):
        heatmap = heatmap + local_w[i] * masks[i]

    heatmap = heatmap - heatmap.min()
    heatmap = heatmap / (heatmap.max() + 1e-6)
    return heatmap


def save_attention_overlay(image: torch.Tensor, heatmap: torch.Tensor, out_path: str | Path) -> None:
    """원본 이미지 위에 히트맵을 오버레이해 저장한다.

    인자:
        image (torch.Tensor): 입력 이미지 텐서.
        heatmap (torch.Tensor): 어텐션 히트맵.
        out_path (str | Path): 저장 경로.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    img = image.detach().cpu()
    if img.shape[0] == 3:
        img = img[0]
    img = (img - img.min()) / (img.max() + 1e-6)

    plt.figure(figsize=(4, 4))
    plt.imshow(img, cmap="gray")
    if heatmap is not None:
        plt.imshow(heatmap.detach().cpu(), cmap="jet", alpha=0.4)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
