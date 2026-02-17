"""HuggingFace Causal LM 기반 디코더 래퍼."""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM


class HFCausalDecoder(nn.Module):
    """프리픽스 임베딩을 지원하도록 Causal LM을 감싼 래퍼."""

    def __init__(
        self,
        name_or_path: str,
        vocab_size: int | None = None,
        local_files_only: bool = False,
        trust_remote_code: bool = False,
        revision: str | None = None,
    ) -> None:
        super().__init__()
        self.model = AutoModelForCausalLM.from_pretrained(
            name_or_path,
            local_files_only=local_files_only,
            trust_remote_code=trust_remote_code,
            revision=revision,
        )
        if vocab_size is not None and int(vocab_size) != int(self.model.config.vocab_size):
            self.model.resize_token_embeddings(int(vocab_size))
        self.hidden_size = int(getattr(self.model.config, "hidden_size", 0))

    def forward(
        self,
        input_ids: torch.Tensor,
        prefix_embeds: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        return_attn: bool = False,
    ):
        tok_embeds = self.model.get_input_embeddings()(input_ids)
        attn_mask = attention_mask
        if prefix_embeds is not None:
            inputs_embeds = torch.cat([prefix_embeds, tok_embeds], dim=1)
            if attention_mask is not None:
                bsz = attention_mask.size(0)
                prefix_len = prefix_embeds.size(1)
                prefix_mask = torch.ones(
                    (bsz, prefix_len),
                    device=attention_mask.device,
                    dtype=attention_mask.dtype,
                )
                attn_mask = torch.cat([prefix_mask, attention_mask], dim=1)
        else:
            inputs_embeds = tok_embeds

        outputs = self.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attn_mask,
            output_attentions=return_attn,
            use_cache=False,
        )
        attn = None
        if return_attn and outputs.attentions:
            attn = outputs.attentions[-1]
        return outputs.logits, attn
