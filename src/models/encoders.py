"""시각 인코더 모듈(ResNet-50 또는 DINOv2)."""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

_DINOV2_NAME_MAP = {
    "dinov2_vits14": "vit_small_patch14_dinov2",
    "dinov2_vitb14": "vit_base_patch14_dinov2",
    "dinov2_vitl14": "vit_large_patch14_dinov2",
    "dinov2_vitg14": "vit_giant_patch14_dinov2",
}


class VisualEncoder(nn.Module):
    """지정된 백본으로부터 글로벌 비전 피처를 추출한다."""

    def __init__(
        self,
        encoder_name: str = "resnet50",
        pretrained: bool = False,
        backend: str = "timm",
    ) -> None:
        """비전 인코더를 초기화한다.

        인자:
            encoder_name (str): 백본 이름(resnet50 또는 timm/dinov2 모델명).
            pretrained (bool): ImageNet 사전학습 가중치 사용 여부.
            backend (str): DINOv2 로딩 방식(timm 또는 torchhub).
        """
        super().__init__()
        self.encoder_name = encoder_name
        self.out_dim = None

        if encoder_name == "resnet50":
            weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
            self.backbone = resnet50(weights=weights)
            self.backbone.fc = nn.Identity()
            self.out_dim = 2048
        else:
            if backend == "torchhub":
                # torch.hub 기반 DINOv2 로딩
                # e.g., encoder_name: dinov2_vitb14
                self.backbone = torch.hub.load(
                    "facebookresearch/dinov2", encoder_name, pretrained=pretrained
                )
                self.out_dim = getattr(self.backbone, "embed_dim", None) or getattr(
                    self.backbone, "num_features", None
                )
            else:
                try:
                    import timm
                except Exception as exc:  # pragma: no cover - 환경 의존
                    raise RuntimeError("timm 설치가 필요합니다.") from exc

                model_name = _DINOV2_NAME_MAP.get(encoder_name, encoder_name)

                def _create(**kwargs):
                    return timm.create_model(
                        model_name,
                        pretrained=pretrained,
                        num_classes=0,
                        global_pool="avg",
                        **kwargs,
                    )

                try:
                    self.backbone = _create()
                except RuntimeError as exc:
                    # dinov2 가중치와 모델 스키마 불일치(fc_norm vs norm) 대응.
                    msg = str(exc)
                    if pretrained and ("fc_norm" in msg and "norm" in msg):
                        try:
                            self.backbone = _create(fc_norm=False)
                        except TypeError:
                            raise exc
                    else:
                        raise
                self.out_dim = getattr(self.backbone, "num_features", None) or getattr(
                    self.backbone, "embed_dim", None
                )
            if self.out_dim is None:
                raise ValueError(f"Cannot infer encoder output dim for {encoder_name}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """이미지 배치로부터 (B,D) 피처를 반환한다.

        인자:
            x (torch.Tensor): 입력 이미지 텐서.

        반환:
            torch.Tensor: 글로벌 피처 텐서.
        """
        out = self.backbone(x)
        if out.dim() == 3:
            # (B, N, D) -> cls token
            return out[:, 0]
        return out
