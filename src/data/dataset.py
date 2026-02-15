"""흉부 X-ray 데이터셋 로딩/전처리와 배치 구성 로직."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from PIL import Image
import pydicom
from torchvision import transforms

from src.data.chexpert_labels import CHEXPERT_LABELS

@dataclass
class TextConfig:
    """텍스트 특수 토큰 설정을 담는 데이터 클래스."""

    bos: str
    eos: str
    pad: str
    findings: str
    impression: str


def build_image_transform() -> transforms.Compose:
    """모델 입력용 이미지 변환 파이프라인을 구성한다.

    반환:
        transforms.Compose: 흑백 변환/리사이즈/정규화를 포함한 변환 파이프라인.
    """
    return transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def load_image(path: Path) -> Image.Image:
    """이미지 파일을 로드하며, DICOM이면 정규화 후 PIL 이미지로 변환한다.

    인자:
        path (Path): 이미지 파일 경로.

    반환:
        Image.Image: 로드된 PIL 이미지(그레이스케일).
    """
    if path.suffix.lower() in {".dcm", ".dicom"}:
        ds = pydicom.dcmread(str(path))
        # DICOM의 픽셀 값을 실수형으로 변환하고 스케일/오프셋 보정 적용.
        img = ds.pixel_array.astype(np.float32)
        if hasattr(ds, "RescaleSlope") and hasattr(ds, "RescaleIntercept"):
            img = img * float(ds.RescaleSlope) + float(ds.RescaleIntercept)
        # MONOCHROME1은 반전된 명암 체계이므로 뒤집어 준다.
        if getattr(ds, "PhotometricInterpretation", "") == "MONOCHROME1":
            img = img.max() - img
        # 0~1 범위로 정규화 후 8비트 이미지로 변환.
        img = img - img.min()
        img = img / (img.max() + 1e-6)
        img = (img * 255.0).astype(np.uint8)
        return Image.fromarray(img)
    return Image.open(path).convert("L")


def build_text_sequence(findings: str, impression: str, tcfg: TextConfig) -> str:
    """Findings/Impression을 특수 토큰과 함께 하나의 시퀀스로 결합한다.

    인자:
        findings (str): Findings 섹션 텍스트.
        impression (str): Impression 섹션 텍스트.
        tcfg (TextConfig): 특수 토큰 설정.

    반환:
        str: 결합된 텍스트 시퀀스.
    """
    findings = findings or ""
    impression = impression or ""
    return f"{tcfg.bos} {tcfg.findings} {findings} {tcfg.impression} {impression} {tcfg.eos}".strip()


def build_prompt_sequence(tcfg: TextConfig) -> str:
    """생성용 프롬프트(<BOS> <IMPRESSION>)를 만든다.

    인자:
        tcfg (TextConfig): 특수 토큰 설정.

    반환:
        str: 프롬프트 문자열.
    """
    return f"{tcfg.bos} {tcfg.impression}".strip()


def tokenize_text(
    text: str,
    tokenizer,
    max_length: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """텍스트를 토큰화하고, 다음 토큰 예측용 라벨(-100 패딩 포함)을 생성한다.

    인자:
        text (str): 입력 텍스트.
        tokenizer: HuggingFace 토크나이저.
        max_length (int): 최대 토큰 길이.

    반환:
        Tuple[torch.Tensor, torch.Tensor]: (input_ids, labels).
    """
    encoded = tokenizer(
        text,
        add_special_tokens=False,
        truncation=True,
        max_length=max_length,
    )
    input_ids = torch.tensor(encoded["input_ids"], dtype=torch.long)
    labels = torch.full_like(input_ids, fill_value=-100)
    if input_ids.numel() > 1:
        labels[:-1] = input_ids[1:]
    return input_ids, labels


def build_loss_weights(
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    tokenizer,
    tcfg: TextConfig,
    w_find: float,
    w_imp: float,
) -> torch.Tensor:
    """Findings/Impression 구간에 서로 다른 가중치를 부여할 loss weight를 만든다.

    인자:
        input_ids (torch.Tensor): 입력 토큰 ID 시퀀스.
        labels (torch.Tensor): 다음 토큰 라벨 시퀀스.
        tokenizer: HuggingFace 토크나이저.
        tcfg (TextConfig): 특수 토큰 설정.
        w_find (float): Findings 구간 가중치.
        w_imp (float): Impression 구간 가중치.

    반환:
        torch.Tensor: 토큰별 손실 가중치 텐서.
    """
    find_id = tokenizer.convert_tokens_to_ids(tcfg.findings)
    imp_id = tokenizer.convert_tokens_to_ids(tcfg.impression)
    bos_id = tokenizer.convert_tokens_to_ids(tcfg.bos)
    eos_id = tokenizer.convert_tokens_to_ids(tcfg.eos)

    weights = torch.zeros_like(labels, dtype=torch.float)
    segment = None
    for i in range(labels.shape[0]):
        if i + 1 >= input_ids.shape[0]:
            break
        token_id = int(input_ids[i + 1])
        if token_id == find_id:
            segment = "find"
            continue
        if token_id == imp_id:
            segment = "imp"
            continue
        if token_id in {bos_id, eos_id}:
            continue
        if labels[i].item() == -100:
            continue
        if segment == "find":
            weights[i] = w_find
        elif segment == "imp":
            weights[i] = w_imp
    return weights


class ChestXrayDataset(Dataset):
    """CSV 기반 흉부 X-ray 데이터셋."""

    def __init__(
        self,
        csv_path: str | Path,
        image_root: str | Path,
        tokenizer,
        text_cfg: TextConfig,
        max_length: int = 256,
        chex_csv_path: Optional[str | Path] = None,
        chex_label_mode: str = "bce",
        split: Optional[str] = None,
        w_find: float = 1.0,
        w_imp: float = 3.0,
    ) -> None:
        """데이터셋을 초기화한다.

        인자:
            csv_path (str | Path): CSV 경로.
            image_root (str | Path): 이미지 루트 디렉터리.
            tokenizer: HuggingFace 토크나이저.
            text_cfg (TextConfig): 특수 토큰 설정.
            max_length (int): 최대 토큰 길이.
            chex_csv_path (str | Path | None): CheXpert 라벨 CSV 경로.
            chex_label_mode (str): "bce" | "ce3" | "ce4".
            split (str | None): split 컬럼이 있을 때 사용할 분할 이름.
            w_find (float): Findings 가중치.
            w_imp (float): Impression 가중치.
        """
        # CSV 로드 및 스플릿 필터링.
        self.df = pd.read_csv(csv_path)
        if split and "split" in self.df.columns:
            self.df = self.df[self.df["split"] == split].reset_index(drop=True)
        self.image_root = Path(image_root)
        self.tokenizer = tokenizer
        self.text_cfg = text_cfg
        self.max_length = max_length
        self.w_find = w_find
        self.w_imp = w_imp
        self.transform = build_image_transform()
        self.chex_label_mode = chex_label_mode

        # CheXpert CSV가 있으면 인덱스 기준으로 라벨을 구성.
        self.chex_labels = None
        if chex_csv_path and Path(chex_csv_path).exists():
            chex_df = pd.read_csv(chex_csv_path)
            if "Reports" in chex_df.columns and "Impression" not in chex_df.columns:
                chex_df = chex_df.rename(columns={"Reports": "Impression"})

            missing_cols = [c for c in CHEXPERT_LABELS if c not in chex_df.columns]
            if missing_cols:
                raise ValueError(f"CheXpert CSV에 누락된 컬럼이 있습니다: {missing_cols}")

            print(f"train rows: {len(self.df)}")
            print(f"chex rows: {len(chex_df)}")
            if len(self.df) != len(chex_df):
                raise ValueError(
                    f"train/chex row 수가 다릅니다: {len(self.df)} vs {len(chex_df)}"
                )

            train_imp = self.df["Impression"].fillna("").astype(str).values
            chex_imp = chex_df["Impression"].fillna("").astype(str).values
            mismatch = train_imp != chex_imp
            mismatch_count = int(mismatch.sum())
            print(f"Impression mismatch count: {mismatch_count}")
            if mismatch_count:
                bad_idx = np.where(mismatch)[0][:5]
                print(f"first mismatch indices: {bad_idx.tolist()}")
                raise ValueError("Impression 순서가 일치하지 않습니다.")

            chex_vals = chex_df[CHEXPERT_LABELS]
            if chex_label_mode == "bce":
                chex_vals = chex_vals.fillna(0).replace(-1.0, 0.5).astype(float)
                self.chex_labels = chex_vals.values
            elif chex_label_mode == "ce3":
                # 0=neg, 1=pos, 2=uncertain(-1)
                chex_vals = chex_vals.replace(-1.0, 2).fillna(0).astype(int)
                self.chex_labels = chex_vals.values
            elif chex_label_mode == "ce4":
                # 0=neg, 1=pos, 2=uncertain(-1), 3=na
                chex_vals = chex_vals.replace(-1.0, 2).fillna(3).astype(int)
                self.chex_labels = chex_vals.values
            else:
                raise ValueError(f"Unsupported chex_label_mode: {chex_label_mode}")

    def __len__(self) -> int:
        """데이터셋 길이를 반환한다.

        반환:
            int: 샘플 개수.
        """
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        """인덱스에 해당하는 샘플을 반환한다.

        인자:
            idx (int): 샘플 인덱스.

        반환:
            dict: 이미지/토큰/라벨/가중치 및 부가 정보가 담긴 샘플.
        """
        row = self.df.iloc[idx]
        image_path = self.image_root / str(row["image"])
        image = self.transform(load_image(image_path))

        findings = row.get("Findings", "")
        impression = row.get("Impression", "")
        # CSV 결측치는 NaN(float)로 들어오므로 문자열로 치환.
        if isinstance(findings, float) and np.isnan(findings):
            findings = ""
        if isinstance(impression, float) and np.isnan(impression):
            impression = ""

        text = build_text_sequence(findings, impression, self.text_cfg)
        input_ids, labels = tokenize_text(text, self.tokenizer, self.max_length)
        weights = build_loss_weights(
            input_ids,
            labels,
            self.tokenizer,
            self.text_cfg,
            self.w_find,
            self.w_imp,
        )

        sample = {
            "image": image,
            "input_ids": input_ids,
            "labels": labels,
            "weights": weights,
            "findings": findings,
            "impression": impression,
        }
        # 캐시된 진단 라벨이 있으면 배치에 포함.
        if self.chex_labels is not None:
            dtype = torch.float if self.chex_label_mode == "bce" else torch.long
            sample["chex_labels"] = torch.tensor(self.chex_labels[idx], dtype=dtype)
        return sample


class SyntheticCXRDataset(Dataset):
    """오프라인 테스트용 합성 데이터셋."""

    def __init__(
        self,
        tokenizer,
        text_cfg: TextConfig,
        num_samples: int = 16,
        max_length: int = 256,
        w_find: float = 1.0,
        w_imp: float = 3.0,
    ) -> None:
        """합성 데이터셋을 초기화한다.

        인자:
            tokenizer: HuggingFace 토크나이저.
            text_cfg (TextConfig): 특수 토큰 설정.
            num_samples (int): 합성 샘플 개수.
            max_length (int): 최대 토큰 길이.
            w_find (float): Findings 가중치.
            w_imp (float): Impression 가중치.
        """
        self.tokenizer = tokenizer
        self.text_cfg = text_cfg
        self.num_samples = num_samples
        self.max_length = max_length
        self.w_find = w_find
        self.w_imp = w_imp
        self.transform = build_image_transform()
        # 간단한 예시 문장을 순환 사용.
        self.samples = [
            (
                "lungs are clear",
                "no acute disease",
            ),
            (
                "heart size is normal",
                "no pneumothorax",
            ),
            (
                "right lung opacity",
                "possible pneumonia",
            ),
            (
                "mild pleural effusion",
                "effusion persists",
            ),
        ]

    def __len__(self) -> int:
        """데이터셋 길이를 반환한다.

        반환:
            int: 샘플 개수.
        """
        return self.num_samples

    def __getitem__(self, idx: int) -> dict:
        """합성 샘플을 생성해 반환한다.

        인자:
            idx (int): 샘플 인덱스.

        반환:
            dict: 이미지/토큰/라벨/가중치 및 합성 진단 라벨이 담긴 샘플.
        """
        findings, impression = self.samples[idx % len(self.samples)]
        image = torch.rand(3, 224, 224)
        text = build_text_sequence(findings, impression, self.text_cfg)
        input_ids, labels = tokenize_text(text, self.tokenizer, self.max_length)
        weights = build_loss_weights(
            input_ids,
            labels,
            self.tokenizer,
            self.text_cfg,
            self.w_find,
            self.w_imp,
        )
        chex_labels = torch.randint(0, 2, (14,), dtype=torch.float)
        return {
            "image": image,
            "input_ids": input_ids,
            "labels": labels,
            "weights": weights,
            "findings": findings,
            "impression": impression,
            "chex_labels": chex_labels,
        }


def collate_fn(batch: List[dict], pad_id: int) -> dict:
    """배치 패딩/마스킹을 수행하는 collate 함수.

    인자:
        batch (List[dict]): 개별 샘플 리스트.
        pad_id (int): 패딩 토큰 ID.

    반환:
        dict: 텐서로 스택된 배치 딕셔너리.
    """
    images = torch.stack([b["image"] for b in batch], dim=0)
    input_ids = pad_sequence([b["input_ids"] for b in batch], batch_first=True, padding_value=pad_id)
    labels = pad_sequence([b["labels"] for b in batch], batch_first=True, padding_value=-100)
    weights = pad_sequence([b["weights"] for b in batch], batch_first=True, padding_value=0.0)
    attention_mask = (input_ids != pad_id).long()

    out = {
        "image": images,
        "input_ids": input_ids,
        "labels": labels,
        "weights": weights,
        "attention_mask": attention_mask,
        "findings": [b.get("findings", "") for b in batch],
        "impression": [b.get("impression", "") for b in batch],
    }
    # chex_labels가 포함된 배치라면 텐서로 합친다.
    if "chex_labels" in batch[0]:
        out["chex_labels"] = torch.stack([b["chex_labels"] for b in batch], dim=0)
    return out
