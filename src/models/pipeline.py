"""세그멘터-인코더-퓨전-디코더를 묶은 전체 파이프라인 모델."""

from __future__ import annotations

import torch
import torch.nn as nn

from .anatomy_segmenter import AnatomySegmenter
from .encoders import VisualEncoder
from .fusion import LocalGlobalFusion
from .diagnosis import DiagnosisHead
from .decoder import DecoderOnlyTransformer


class LungImpressionModel(nn.Module):
    """흉부 X-ray Impression 생성을 위한 통합 모델."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        n_heads: int = 8,
        n_layers: int = 6,
        dropout: float = 0.1,
        diag_classes: int = 14,
        n_visual_tokens: int = 4,
        n_diag_tokens: int = 2,
        visual_encoder: str = "resnet50",
        visual_encoder_backend: str = "timm",
        visual_pretrained: bool = False,
        freeze_encoder: bool = False,
        diag_loss_type: str = "bce",
        segmentation_cfg: dict | None = None,
    ) -> None:
        """모델을 초기화한다.

        인자:
            vocab_size (int): 어휘 크기.
            d_model (int): 디코더 차원.
            n_heads (int): 디코더 헤드 개수.
            n_layers (int): 디코더 레이어 개수.
            dropout (float): 드롭아웃 비율.
            diag_classes (int): 진단 라벨 개수.
            n_visual_tokens (int): 시각 프리픽스 토큰 개수.
            n_diag_tokens (int): 진단 프리픽스 토큰 개수.
            visual_encoder (str): 시각 인코더 백본 이름.
            visual_encoder_backend (str): 시각 인코더 로딩 방식(timm/torchhub).
            visual_pretrained (bool): 비전 인코더 사전학습 여부.
            freeze_encoder (bool): 시각 인코더 동결 여부.
            diag_loss_type (str): "bce" | "ce3" | "ce4".
            segmentation_cfg (dict | None): 세그멘터 설정.
        """
        super().__init__()
        segmentation_cfg = segmentation_cfg or {}
        # 해부학 세그멘터는 추론 전용이며, 가중치가 없으면 더미 마스크를 사용.
        self.segmenter = AnatomySegmenter(
            model_path=segmentation_cfg.get("model_path"),
            hf_model_id=segmentation_cfg.get("hf_model_id"),
            hf_trust_remote_code=segmentation_cfg.get("hf_trust_remote_code", True),
            hf_kwargs=segmentation_cfg.get("hf_kwargs"),
            use_dummy_if_missing=segmentation_cfg.get("use_dummy_if_missing", True),
            device=segmentation_cfg.get("device", "cpu"),
        )
        self.local_view_mode = segmentation_cfg.get("local_view_mode", "bbox")
        self.diag_loss_type = diag_loss_type
        self.encoder = VisualEncoder(
            encoder_name=visual_encoder,
            pretrained=visual_pretrained,
            backend=visual_encoder_backend,
        )
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
        visual_dim = self.encoder.out_dim
        # 퓨전은 시각 인코더 피처 차원에 대해 작동.
        self.fusion = LocalGlobalFusion(d_model=visual_dim, n_heads=8)
        num_diag_classes = 1
        if diag_loss_type == "ce3":
            num_diag_classes = 3
        elif diag_loss_type == "ce4":
            num_diag_classes = 4
        elif diag_loss_type != "bce":
            raise ValueError(f"Unsupported diag_loss_type: {diag_loss_type}")

        self.diagnosis = DiagnosisHead(
            d_model=visual_dim,
            num_labels=diag_classes,
            num_classes=num_diag_classes,
        )

        # 시각/진단 피처를 디코더 토큰 공간으로 투영.
        self.visual_proj = nn.Linear(visual_dim, d_model)
        self.diag_proj = nn.Linear(diag_classes, n_diag_tokens * d_model)

        self.decoder = DecoderOnlyTransformer(
            vocab_size=vocab_size,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            dropout=dropout,
            max_len=512,
        )

        self.n_visual_tokens = n_visual_tokens
        self.n_diag_tokens = n_diag_tokens

    def encode_visual(self, images: torch.Tensor) -> dict:
        """세그멘터+인코더+퓨전을 통해 시각 피처/진단 로짓을 계산한다.

        인자:
            images (torch.Tensor): 입력 이미지 텐서.

        반환:
            dict: 마스크/피처/진단 로짓을 포함한 결과.
        """
        masks = self.segmenter(images)
        b, _, h, w = images.shape
        # 해상도 유지 + bbox 크롭 기반 로컬 뷰 생성.
        local_images = self.build_local_views(images, masks, mode=self.local_view_mode)
        local_images = local_images.view(b * 3, images.shape[1], h, w)

        global_feat = self.encoder(images)
        local_feat = self.encoder(local_images).view(b, 3, -1)
        fused = self.fusion(global_feat, local_feat)
        diag_logits = self.diagnosis(fused)
        return {
            "masks": masks,
            "global_feat": global_feat,
            "local_feat": local_feat,
            "fused": fused,
            "diag_logits": diag_logits,
        }

    def build_local_views(
        self,
        images: torch.Tensor,
        masks: torch.Tensor,
        mode: str | None = None,
    ) -> torch.Tensor:
        """마스크 기반 로컬 뷰를 생성한다.

        인자:
            images (torch.Tensor): (B,C,H,W) 입력 이미지.
            masks (torch.Tensor): (B,3,H,W) 마스크 텐서.
            mode (str | None): "mask" 또는 "bbox".

        반환:
            torch.Tensor: (B,3,C,H,W) 로컬 뷰 텐서.
                - mode="mask": 원본 해상도 유지, 배경 0 마스킹
                - mode="bbox": 마스크 bbox 영역만 크롭 후 원본 해상도로 리사이즈
        """
        mode = mode or self.local_view_mode
        if mode == "mask":
            return images.unsqueeze(1) * masks.unsqueeze(2)
        if mode != "bbox":
            raise ValueError(f"Unsupported local view mode: {mode}")

        b, c, h, w = images.shape
        local = images.new_zeros((b, 3, c, h, w))
        for bi in range(b):
            for mi in range(3):
                mask = masks[bi, mi] > 0
                if not mask.any().item():
                    continue
                ys, xs = torch.nonzero(mask, as_tuple=True)
                y0, y1 = ys.min().item(), ys.max().item() + 1
                x0, x1 = xs.min().item(), xs.max().item() + 1
                crop = images[bi : bi + 1, :, y0:y1, x0:x1]
                resized = torch.nn.functional.interpolate(
                    crop,
                    size=(h, w),
                    mode="bilinear",
                    align_corners=False,
                )
                local[bi, mi] = resized[0]
        return local

    def build_prefix(self, global_feat: torch.Tensor, local_feat: torch.Tensor, diag_logits: torch.Tensor) -> torch.Tensor:
        """시각/진단 피처를 디코더 프리픽스 토큰으로 구성한다.

        인자:
            global_feat (torch.Tensor): 글로벌 피처.
            local_feat (torch.Tensor): 로컬 피처.
            diag_logits (torch.Tensor): 진단 로짓.

        반환:
            torch.Tensor: 프리픽스 임베딩.
        """
        # 시각 토큰 순서: [global, left, right, heart]
        global_tok = self.visual_proj(global_feat).unsqueeze(1)
        local_tok = self.visual_proj(local_feat)  # 로컬 토큰 (B,3,D)
        visual_tokens = torch.cat([global_tok, local_tok], dim=1)
        if visual_tokens.shape[1] != self.n_visual_tokens:
            raise ValueError("n_visual_tokens must be 4 (global + 3 local).")

        # 진단 로짓을 소수의 프리픽스 토큰으로 변환.
        if self.diag_loss_type == "bce":
            diag_probs = torch.sigmoid(diag_logits)
        else:
            probs = torch.softmax(diag_logits, dim=-1)
            # class index: 0=neg, 1=pos, 2=uncertain, 3=na(ce4)
            pos = probs[..., 1]
            if probs.shape[-1] >= 3:
                pos = pos + 0.5 * probs[..., 2]
            diag_probs = pos
        diag_tokens = self.diag_proj(diag_probs).view(diag_probs.shape[0], self.n_diag_tokens, -1)
        prefix = torch.cat([visual_tokens, diag_tokens], dim=1)
        return prefix

    def forward(
        self,
        images: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        return_attn: bool = False,
    ) -> dict:
        """학습/평가용 forward: 로짓, 진단 로짓, 마스크, 어텐션을 반환한다.

        인자:
            images (torch.Tensor): 입력 이미지 텐서.
            input_ids (torch.Tensor): 입력 토큰 ID.
            attention_mask (torch.Tensor | None): 패딩 마스크.
            return_attn (bool): 어텐션 반환 여부.

        반환:
            dict: 로짓/진단 로짓/마스크/어텐션 등의 결과.
        """
        enc = self.encode_visual(images)
        prefix = self.build_prefix(enc["global_feat"], enc["local_feat"], enc["diag_logits"])
        logits, attn = self.decoder(
            input_ids,
            prefix_embeds=prefix,
            attention_mask=attention_mask,
            return_attn=return_attn,
        )
        return {
            "logits": logits,
            "diag_logits": enc["diag_logits"],
            "masks": enc["masks"],
            "attn": attn,
            "prefix_len": prefix.shape[1],
        }

    @torch.no_grad()
    def generate(
        self,
        images: torch.Tensor,
        tokenizer,
        max_len: int = 128,
        device: str = "cpu",
        min_len: int = 0,
        do_sample: bool = False,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
        repetition_penalty: float = 1.0,
        no_repeat_ngram_size: int = 0,
        stop_on_eos: bool = True,
        prompt_mode: str = "impression_only",
    ) -> dict:
        """<BOS><IMPRESSION> 프롬프트로 Impression을 autoregressive하게 생성한다.

        인자:
            images (torch.Tensor): 입력 이미지 텐서.
            tokenizer: 토크나이저.
            max_len (int): 생성 최대 길이.
            device (str): 디바이스 문자열.

        반환:
            dict: 생성 텍스트와 어텐션/마스크 정보.
        """
        self.eval()
        enc = self.encode_visual(images)
        prefix = self.build_prefix(enc["global_feat"], enc["local_feat"], enc["diag_logits"])
        prefix_len = prefix.shape[1]

        bos_id = tokenizer.bos_token_id
        if bos_id is None or bos_id < 0:
            bos_id = tokenizer.convert_tokens_to_ids("<BOS>")
        find_id = tokenizer.convert_tokens_to_ids("<FINDINGS>")
        imp_id = tokenizer.convert_tokens_to_ids("<IMPRESSION>")
        eos_id = tokenizer.eos_token_id
        if eos_id is None or eos_id < 0:
            eos_id = tokenizer.convert_tokens_to_ids("<EOS>")

        # 프롬프트 구성.
        mode = str(prompt_mode or "impression_only").lower()
        if mode == "findings_impression":
            prompt_ids = [bos_id, find_id, imp_id]
        else:
            prompt_ids = [bos_id, imp_id]

        # 프롬프트로 시작해 오토리그레시브 디코딩 수행.
        input_ids = torch.tensor([prompt_ids], device=device, dtype=torch.long)
        attn_per_step = []

        for _ in range(max_len):
            logits, attn = self.decoder(input_ids, prefix_embeds=prefix, return_attn=True)
            next_token_logits = logits[:, -1, :]

            if repetition_penalty and repetition_penalty != 1.0:
                for batch_idx in range(next_token_logits.size(0)):
                    used = set(input_ids[batch_idx].tolist())
                    for token_id in used:
                        if next_token_logits[batch_idx, token_id] < 0:
                            next_token_logits[batch_idx, token_id] *= repetition_penalty
                        else:
                            next_token_logits[batch_idx, token_id] /= repetition_penalty

            if no_repeat_ngram_size and no_repeat_ngram_size > 1:
                banned_tokens = []
                for seq in input_ids.tolist():
                    if len(seq) < no_repeat_ngram_size:
                        banned_tokens.append([])
                        continue
                    ngrams = {}
                    for i in range(len(seq) - no_repeat_ngram_size + 1):
                        prefix_ng = tuple(seq[i : i + no_repeat_ngram_size - 1])
                        next_tok = seq[i + no_repeat_ngram_size - 1]
                        ngrams.setdefault(prefix_ng, set()).add(next_tok)
                    current = tuple(seq[-(no_repeat_ngram_size - 1) :])
                    banned_tokens.append(list(ngrams.get(current, [])))
                for batch_idx, banned in enumerate(banned_tokens):
                    if banned:
                        next_token_logits[batch_idx, banned] = -float("inf")

            prompt_len = len(prompt_ids)
            if eos_id is not None and eos_id >= 0 and (input_ids.shape[1] - prompt_len) < int(min_len):
                next_token_logits[:, eos_id] = -float("inf")

            if temperature and temperature != 1.0:
                temp = float(temperature)
                if temp <= 0:
                    temp = 1.0
                next_token_logits = next_token_logits / temp

            if top_k and top_k > 0:
                k = min(int(top_k), next_token_logits.size(-1))
                kth_vals = torch.topk(next_token_logits, k)[0][..., -1, None]
                next_token_logits = next_token_logits.masked_fill(next_token_logits < kth_vals, -float("inf"))

            if top_p and top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                probs = torch.softmax(sorted_logits, dim=-1)
                cumprobs = torch.cumsum(probs, dim=-1)
                sorted_remove = cumprobs > float(top_p)
                sorted_remove[..., 0] = 0
                indices_to_remove = sorted_remove.scatter(1, sorted_indices, sorted_remove)
                next_token_logits = next_token_logits.masked_fill(indices_to_remove, -float("inf"))

            if do_sample:
                probs = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            if attn is not None:
                # 그라운딩 분석을 위해 프리픽스 토큰에 대한 어텐션을 기록.
                step_attn = attn[:, :, -1, :prefix_len].mean(dim=1)
                attn_per_step.append(step_attn.cpu())
            input_ids = torch.cat([input_ids, next_token], dim=1)
            if stop_on_eos and eos_id is not None and eos_id >= 0:
                if (next_token == eos_id).any().item():
                    break

        generated = input_ids[:, len(prompt_ids) :]  # 프롬프트 토큰 제거
        text = tokenizer.batch_decode(generated, skip_special_tokens=True)[0]
        attn_tokens = torch.cat(attn_per_step, dim=0) if attn_per_step else None

        return {
            "text": text.strip(),
            "attn_tokens": attn_tokens,
            "masks": enc["masks"].detach().cpu(),
        }
