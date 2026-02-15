"""어텐션 기반 그라운딩 평가 로직."""

from __future__ import annotations

import numpy as np
import torch

from .attention import attention_to_heatmap


def mask_topk(image: torch.Tensor, heatmap: torch.Tensor, k: float = 0.1) -> torch.Tensor:
    """히트맵 상위 k 비율 영역을 마스킹한다.

    인자:
        image (torch.Tensor): 입력 이미지 텐서.
        heatmap (torch.Tensor): 히트맵 텐서.
        k (float): 마스킹 비율(0~1).

    반환:
        torch.Tensor: 마스킹된 이미지 텐서.
    """
    if heatmap is None:
        return image
    flat = heatmap.flatten()
    thresh = torch.quantile(flat, 1 - k)
    mask = heatmap >= thresh
    masked = image.clone()
    masked[:, mask] = 0.0
    return masked


def grounding_eval(
    model,
    image: torch.Tensor,
    tokenizer,
    labeler,
    k: float = 0.1,
    device: str = "cpu",
) -> dict:
    """원본/마스킹 이미지를 비교해 텍스트/라벨 변화량을 측정한다.

    인자:
        model: 생성 모델.
        image (torch.Tensor): 입력 이미지 텐서.
        tokenizer: 토크나이저.
        labeler: 라벨러(또는 None).
        k (float): 마스킹 비율.
        device (str): 디바이스 문자열.

    반환:
        dict: 생성 텍스트, 마스킹 텍스트, 변화율, 히트맵 등을 포함한 결과.
    """
    model.eval()
    with torch.no_grad():
        out = model.generate(image.unsqueeze(0).to(device), tokenizer, device=device)
        heatmap = attention_to_heatmap(out["attn_tokens"], out["masks"])
        masked_image = mask_topk(image, heatmap, k=k)
        out_masked = model.generate(masked_image.unsqueeze(0).to(device), tokenizer, device=device)

    original_text = out["text"]
    masked_text = out_masked["text"]

    labels_before = labeler([original_text]) if labeler is not None else None
    labels_after = labeler([masked_text]) if labeler is not None else None

    change = None
    if labels_before is not None and labels_after is not None:
        change = (np.array(labels_before) != np.array(labels_after)).astype(int).mean()

    return {
        "original_text": original_text,
        "masked_text": masked_text,
        "label_change_rate": float(change) if change is not None else 0.0,
        "heatmap": heatmap,
    }
