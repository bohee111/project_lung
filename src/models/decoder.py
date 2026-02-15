"""프리픽스 토큰을 지원하는 디코더-온리 Transformer."""

from __future__ import annotations

import math
import torch
import torch.nn as nn


def make_causal_mask(size: int, device: torch.device) -> torch.Tensor:
    """미래 토큰을 차단하는 상삼각 마스크를 만든다.

    인자:
        size (int): 시퀀스 길이.
        device (torch.device): 텐서를 생성할 디바이스.

    반환:
        torch.Tensor: (size, size) 형태의 마스크.
    """
    mask = torch.full((size, size), float("-inf"), device=device)
    mask = torch.triu(mask, diagonal=1)
    return mask


class MultiHeadSelfAttention(nn.Module):
    """단순화된 멀티헤드 자기 어텐션 레이어."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1) -> None:
        """멀티헤드 자기 어텐션을 초기화한다.

        인자:
            d_model (int): 모델 차원.
            n_heads (int): 헤드 개수.
            dropout (float): 드롭아웃 비율.
        """
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.qkv = nn.Linear(d_model, d_model * 3)
        self.out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        key_padding_mask: torch.Tensor | None = None,
        need_weights: bool = False,
    ):
        """입력 시퀀스에 대해 어텐션 결과와 (옵션) 가중치를 반환한다.

        인자:
            x (torch.Tensor): 입력 시퀀스 텐서.
            attn_mask (torch.Tensor | None): 어텐션 마스크.
            key_padding_mask (torch.Tensor | None): 패딩 마스크.
            need_weights (bool): 어텐션 가중치 반환 여부.

        반환:
            Tuple[torch.Tensor, torch.Tensor | None]: (출력, 어텐션 가중치).
        """
        b, t, d = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(b, t, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(b, t, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(b, t, self.n_heads, self.d_head).transpose(1, 2)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_head)
        if attn_mask is not None:
            attn_scores = attn_scores + attn_mask
        if key_padding_mask is not None:
            mask = key_padding_mask.unsqueeze(1).unsqueeze(2)  # 패딩 마스크 형태 (B,1,1,T)
            attn_scores = attn_scores.masked_fill(mask, float("-inf"))

        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        out = torch.matmul(attn_weights, v)
        out = out.transpose(1, 2).contiguous().view(b, t, d)
        out = self.out(out)
        if need_weights:
            return out, attn_weights
        return out, None


class FeedForward(nn.Module):
    """Transformer 블록의 FFN 서브레이어."""

    def __init__(self, d_model: int, dropout: float = 0.1) -> None:
        """FFN 서브레이어를 초기화한다.

        인자:
            d_model (int): 모델 차원.
            dropout (float): 드롭아웃 비율.
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """비선형 FFN 변환을 적용한다.

        인자:
            x (torch.Tensor): 입력 텐서.

        반환:
            torch.Tensor: 출력 텐서.
        """
        return self.net(x)


class DecoderBlock(nn.Module):
    """LayerNorm + Self-Attn + FFN으로 구성된 디코더 블록."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1) -> None:
        """디코더 블록을 초기화한다.

        인자:
            d_model (int): 모델 차원.
            n_heads (int): 헤드 개수.
            dropout (float): 드롭아웃 비율.
        """
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadSelfAttention(d_model, n_heads, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model, dropout)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        key_padding_mask: torch.Tensor | None = None,
        need_weights: bool = False,
    ):
        """잔차 연결을 포함한 한 블록의 출력을 계산한다.

        인자:
            x (torch.Tensor): 입력 텐서.
            attn_mask (torch.Tensor | None): 어텐션 마스크.
            key_padding_mask (torch.Tensor | None): 패딩 마스크.
            need_weights (bool): 어텐션 가중치 반환 여부.

        반환:
            Tuple[torch.Tensor, torch.Tensor | None]: (출력, 어텐션 가중치).
        """
        attn_out, attn_weights = self.attn(
            self.ln1(x),
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=need_weights,
        )
        x = x + attn_out
        x = x + self.ff(self.ln2(x))
        return x, attn_weights


class DecoderOnlyTransformer(nn.Module):
    """프리픽스 임베딩을 지원하는 디코더-온리 Transformer."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        n_heads: int = 8,
        n_layers: int = 6,
        max_len: int = 512,
        dropout: float = 0.1,
    ) -> None:
        """디코더-온리 Transformer를 초기화한다.

        인자:
            vocab_size (int): 어휘 크기.
            d_model (int): 모델 차원.
            n_heads (int): 헤드 개수.
            n_layers (int): 레이어 개수.
            max_len (int): 최대 시퀀스 길이.
            dropout (float): 드롭아웃 비율.
        """
        super().__init__()
        self.d_model = d_model
        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(max_len, d_model)
        self.blocks = nn.ModuleList(
            [DecoderBlock(d_model, n_heads, dropout) for _ in range(n_layers)]
        )
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        prefix_embeds: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        return_attn: bool = False,
    ):
        """토큰+프리픽스를 결합해 로짓과 (옵션) 어텐션을 반환한다.

        인자:
            input_ids (torch.Tensor): 입력 토큰 ID.
            prefix_embeds (torch.Tensor | None): 프리픽스 임베딩.
            attention_mask (torch.Tensor | None): 패딩 마스크.
            return_attn (bool): 어텐션 가중치 반환 여부.

        반환:
            Tuple[torch.Tensor, torch.Tensor | None]: (로짓, 어텐션 가중치).
        """
        b, t = input_ids.shape
        prefix_len = 0 if prefix_embeds is None else prefix_embeds.shape[1]
        device = input_ids.device

        tok_emb = self.token_embed(input_ids)
        positions = torch.arange(prefix_len, prefix_len + t, device=device).unsqueeze(0)
        tok_emb = tok_emb + self.pos_embed(positions)

        if prefix_embeds is not None:
            prefix_positions = torch.arange(0, prefix_len, device=device).unsqueeze(0)
            prefix_embeds = prefix_embeds + self.pos_embed(prefix_positions)
            x = torch.cat([prefix_embeds, tok_emb], dim=1)
        else:
            x = tok_emb

        seq_len = x.shape[1]
        attn_mask = make_causal_mask(seq_len, device)

        key_padding_mask = None
        if attention_mask is not None:
            if prefix_len > 0:
                prefix_mask = torch.ones((b, prefix_len), device=device, dtype=attention_mask.dtype)
                attn_pad = torch.cat([prefix_mask, attention_mask], dim=1)
            else:
                attn_pad = attention_mask
            key_padding_mask = attn_pad == 0

        attn_weights = None
        for i, block in enumerate(self.blocks):
            x, weights = block(
                x,
                attn_mask=attn_mask,
                key_padding_mask=key_padding_mask,
                need_weights=return_attn and (i == len(self.blocks) - 1),
            )
            if weights is not None:
                attn_weights = weights

        x = self.ln_f(x)
        logits = self.head(x)
        return logits, attn_weights
