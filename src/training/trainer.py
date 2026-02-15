"""학습 루프 및 체크포인트 저장 로직."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .loss import weighted_ce_loss, diagnosis_loss


class Trainer:
    """모델 학습과 체크포인트 저장을 담당한다."""

    def __init__(
        self,
        model,
        optimizer,
        device: torch.device,
        lambda_diagnosis: float = 0.5,
        diag_loss_type: str = "bce",
        amp_enabled: bool = False,
        amp_dtype: str = "bf16",
    ) -> None:
        """Trainer를 초기화한다.

        인자:
            model: 학습할 모델.
            optimizer: 최적화기.
            device (torch.device): 학습 디바이스.
            lambda_diagnosis (float): 진단 손실 가중치.
        """
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.lambda_diagnosis = lambda_diagnosis
        self.diag_loss_type = diag_loss_type
        self.amp_enabled = bool(amp_enabled) and self.device.type == "cuda"
        self.amp_dtype = self._resolve_amp_dtype(amp_dtype)
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.amp_enabled and self.amp_dtype == torch.float16)

    @staticmethod
    def _resolve_amp_dtype(amp_dtype: str) -> torch.dtype:
        dtype = str(amp_dtype).lower()
        if dtype in {"bf16", "bfloat16"}:
            return torch.bfloat16
        if dtype in {"fp16", "float16"}:
            return torch.float16
        raise ValueError(f"Unsupported amp_dtype: {amp_dtype}")

    def train_epoch(self, loader: DataLoader, log_every: int = 10) -> Dict[str, float]:
        """한 epoch 동안 학습하고 평균 손실을 반환한다.

        인자:
            loader (DataLoader): 학습 데이터 로더.
            log_every (int): 로그 출력 주기(0이면 출력 안 함).

        반환:
            Dict[str, float]: 평균 손실 지표.
        """
        self.model.train()
        total_loss = 0.0
        total_gen = 0.0
        total_diag = 0.0
        steps = 0

        for batch in tqdm(loader, desc="train", leave=False):
            images = batch["image"].to(self.device)
            input_ids = batch["input_ids"].to(self.device)
            labels = batch["labels"].to(self.device)
            weights = batch["weights"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            diag_labels = batch.get("chex_labels")
            if diag_labels is not None:
                diag_labels = diag_labels.to(self.device)

            with torch.autocast(
                device_type=self.device.type,
                dtype=self.amp_dtype,
                enabled=self.amp_enabled,
            ):
                outputs = self.model(
                    images=images,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )
                logits = outputs["logits"]
                diag_logits = outputs["diag_logits"]

                # 생성 손실: 가중치 CE
                gen_loss = weighted_ce_loss(logits[:, -labels.shape[1] :, :], labels, weights)
                loss = gen_loss

                diag_loss_val = torch.tensor(0.0, device=self.device)
                if diag_labels is not None:
                    # 진단 라벨 손실을 보조로 추가.
                    diag_loss_val = diagnosis_loss(diag_logits, diag_labels, loss_type=self.diag_loss_type)
                    loss = loss + self.lambda_diagnosis * diag_loss_val

            self.optimizer.zero_grad()
            if self.scaler.is_enabled():
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                self.optimizer.step()

            total_loss += loss.item()
            total_gen += gen_loss.item()
            total_diag += diag_loss_val.item()
            steps += 1

            if log_every and steps % log_every == 0:
                avg = total_loss / steps
                print(f"step {steps}: loss={avg:.4f}")

        return {
            "loss": total_loss / max(steps, 1),
            "gen_loss": total_gen / max(steps, 1),
            "diag_loss": total_diag / max(steps, 1),
        }

    def save_checkpoint(self, path: str | Path, epoch: int) -> None:
        """모델 state_dict와 epoch 정보를 저장한다.

        인자:
            path (str | Path): 저장 경로.
            epoch (int): 현재 epoch 번호.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"epoch": epoch, "state_dict": self.model.state_dict()}, path)

    @torch.no_grad()
    def eval_epoch(self, loader: DataLoader) -> Dict[str, float]:
        """평가 데이터셋에 대한 손실을 계산한다.

        인자:
            loader (DataLoader): 평가 데이터 로더.

        반환:
            Dict[str, float]: 평균 손실 지표.
        """
        self.model.eval()
        total_loss = 0.0
        total_gen = 0.0
        total_diag = 0.0
        steps = 0

        for batch in tqdm(loader, desc="eval-loss", leave=False):
            images = batch["image"].to(self.device)
            input_ids = batch["input_ids"].to(self.device)
            labels = batch["labels"].to(self.device)
            weights = batch["weights"].to(self.device)
            attention_mask = batch.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.device)
            diag_labels = batch.get("chex_labels")
            if diag_labels is not None:
                diag_labels = diag_labels.to(self.device)

            with torch.autocast(
                device_type=self.device.type,
                dtype=self.amp_dtype,
                enabled=self.amp_enabled,
            ):
                outputs = self.model(
                    images=images,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )
                logits = outputs["logits"]
                diag_logits = outputs["diag_logits"]

                gen_loss = weighted_ce_loss(logits[:, -labels.shape[1] :, :], labels, weights)
                diag_loss_val = torch.tensor(0.0, device=self.device)
                if diag_labels is not None:
                    diag_loss_val = diagnosis_loss(diag_logits, diag_labels, loss_type=self.diag_loss_type)
                loss = gen_loss + self.lambda_diagnosis * diag_loss_val

            total_loss += loss.item()
            total_gen += gen_loss.item()
            total_diag += diag_loss_val.item()
            steps += 1

        return {
            "loss": total_loss / max(steps, 1),
            "gen_loss": total_gen / max(steps, 1),
            "diag_loss": total_diag / max(steps, 1),
        }
