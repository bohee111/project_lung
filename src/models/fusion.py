"""글로벌/로컬 비전 피처를 결합하는 어텐션 기반 퓨전 모듈."""

from __future__ import annotations

import torch
import torch.nn as nn


class LocalGlobalFusion(nn.Module):
    """글로벌 피처를 쿼리로 하여 로컬 피처에 대한 어텐션을 수행한다."""

    def __init__(self, d_model: int, n_heads: int) -> None:
        """퓨전 모듈을 초기화한다.

        인자:
            d_model (int): 피처 차원.
            n_heads (int): 멀티헤드 개수.
        """
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, global_feat: torch.Tensor, local_feats: torch.Tensor) -> torch.Tensor:
        """(B,D) 글로벌 피처와 (B,3,D) 로컬 피처를 융합해 (B,D) 반환.

        인자:
            global_feat (torch.Tensor): 글로벌 피처.
            local_feats (torch.Tensor): 로컬 피처들.

        반환:
            torch.Tensor: 융합된 피처.
        """
        # global_feat는 (B, D), local_feats는 (B, 3, D)
        query = global_feat.unsqueeze(1)
        attn_out, _ = self.attn(query, local_feats, local_feats, need_weights=False)
        fused = self.norm(attn_out.squeeze(1) + global_feat)
        return fused
