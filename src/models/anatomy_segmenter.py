"""해부학 영역 세그멘터(실제 모델 또는 더미 마스크)."""

from __future__ import annotations

from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# 확인용
class DummySegmenter(nn.Module):
    """간단한 타원 마스크로 좌/우 폐와 심장 영역을 근사하는 더미 세그멘터."""

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """입력 이미지 해상도에 맞춘 (B,3,H,W) 마스크를 반환한다.

        인자:
            images (torch.Tensor): 입력 이미지 텐서.

        반환:
            torch.Tensor: 좌/우 폐와 심장 마스크.
        """
        # images 텐서: (B, C, H, W)
        b, _, h, w = images.shape
        device = images.device
        ys = torch.linspace(-1, 1, h, device=device).view(1, h, 1)
        xs = torch.linspace(-1, 1, w, device=device).view(1, 1, w)

        left_mask = ((xs + 0.4) ** 2 / 0.4**2 + (ys) ** 2 / 0.7**2) <= 1.0
        right_mask = ((xs - 0.4) ** 2 / 0.4**2 + (ys) ** 2 / 0.7**2) <= 1.0
        heart_mask = ((xs) ** 2 / 0.25**2 + (ys + 0.1) ** 2 / 0.35**2) <= 1.0

        # 마스크는 (H, W) 형태로 맞춘 뒤 (3, H, W)로 스택한다.
        left_mask = left_mask.squeeze(0)
        right_mask = right_mask.squeeze(0)
        heart_mask = heart_mask.squeeze(0)
        masks = torch.stack(
            [left_mask.float(), right_mask.float(), heart_mask.float()], dim=0
        )
        masks = masks.unsqueeze(0).repeat(b, 1, 1, 1)
        return masks


class AnatomySegmenter(nn.Module):
    """세그멘터 로더: TorchScript/일반 체크포인트를 읽거나 더미로 대체."""

    def __init__(
        self,
        model_path: str | None = None,
        hf_model_id: str | None = None,
        hf_trust_remote_code: bool = True,
        hf_kwargs: dict | None = None,
        use_dummy_if_missing: bool = True,
        device: str = "cpu",
    ) -> None:
        """세그멘터를 초기화한다.

        인자:
            model_path (str | None): 모델 경로(있으면 로드).
            hf_model_id (str | None): Hugging Face 모델 ID.
            hf_trust_remote_code (bool): HF 원격 코드 사용 여부.
            hf_kwargs (dict | None): HF 로더에 전달할 추가 인자.
            use_dummy_if_missing (bool): 모델 없을 때 더미 사용 여부.
            device (str): 디바이스 문자열.
        """
        super().__init__()
        self.device = device
        self.model = None
        self.hf_model = None
        self.hf_model_id = hf_model_id
        self.hf_trust_remote_code = hf_trust_remote_code
        self.hf_kwargs = hf_kwargs or {}

        if hf_model_id:
            try:
                from transformers import AutoModel
            except Exception as exc:  # pragma: no cover - 환경 의존
                if not use_dummy_if_missing:
                    raise RuntimeError("transformers 설치가 필요합니다.") from exc
            else:
                try:
                    self.hf_model = AutoModel.from_pretrained(
                        hf_model_id,
                        trust_remote_code=hf_trust_remote_code,
                        **self.hf_kwargs,
                    )
                    self.hf_model.eval().to(device)
                except Exception as exc:  # pragma: no cover - 네트워크/캐시 의존
                    if not use_dummy_if_missing:
                        raise

        if self.hf_model is None and model_path:
            path = Path(model_path)
            if path.exists():
                try:
                    self.model = torch.jit.load(str(path), map_location=device)
                except Exception:
                    self.model = torch.load(str(path), map_location=device)
        if self.model is None and use_dummy_if_missing:
            self.model = DummySegmenter()

    def _to_grayscale_uint8(self, images: torch.Tensor) -> np.ndarray:
        """입력 이미지를 HF 세그멘터 전처리용 grayscale uint8로 변환."""
        img = images.detach().float()
        # 입력 범위 판단: 0~1 또는 정규화된 값 또는 0~255
        max_val = float(img.max())
        min_val = float(img.min())
        if max_val > 2.0:
            img = img / 255.0
        elif min_val < 0.0 or max_val > 1.5:
            if img.shape[1] == 3:
                mean = torch.tensor([0.485, 0.456, 0.406], device=img.device).view(1, 3, 1, 1)
                std = torch.tensor([0.229, 0.224, 0.225], device=img.device).view(1, 3, 1, 1)
                img = img * std + mean
            else:
                img = (img - img.min()) / (img.max() - img.min() + 1e-6)
        img = img.clamp(0.0, 1.0)
        if img.shape[1] == 3:
            gray = img.mean(dim=1)
        else:
            gray = img.squeeze(1)
        gray = (gray * 255.0).round().to(torch.uint8)
        return gray.cpu().numpy()

    def _forward_hf(self, images: torch.Tensor) -> torch.Tensor:
        """HF 세그멘터로 마스크를 예측한다."""
        if self.hf_model is None:
            raise RuntimeError("HF 모델이 로드되지 않았습니다.")
        device = torch.device(self.device)
        np_images = self._to_grayscale_uint8(images)
        preprocessed = [self.hf_model.preprocess(img) for img in np_images]
        x = torch.from_numpy(np.stack(preprocessed)).unsqueeze(1).float().to(device)
        with torch.inference_mode():
            out = self.hf_model(x)
        if "mask" not in out:
            raise RuntimeError("HF 모델 출력에 'mask' 키가 없습니다.")
        mask_logits = out["mask"]
        cls = torch.argmax(mask_logits, dim=1)  # (B,H,W)
        h, w = images.shape[-2:]
        if cls.shape[-2:] != (h, w):
            cls = F.interpolate(cls.unsqueeze(1).float(), size=(h, w), mode="nearest").squeeze(1).long()
        masks = torch.stack([(cls == 1), (cls == 2), (cls == 3)], dim=1).float()
        return masks.to(images.device)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """세그멘테이션 마스크를 반환한다.

        인자:
            images (torch.Tensor): 입력 이미지 텐서.

        반환:
            torch.Tensor: 마스크 텐서.
        """
        if self.hf_model is not None:
            return self._forward_hf(images)
        if self.model is None:
            raise RuntimeError("No segmentation model available and dummy disabled.")
        return self.model(images)
