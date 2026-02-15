"""모델 생성 결과를 정량 평가하는 루틴."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional
from contextlib import contextmanager

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .metrics import (
    compute_bleu,
    compute_rouge_l,
    compute_f1,
    compute_f1_multiclass,
    compute_bertscore,
)
from src.training.loss import weighted_ce_loss, diagnosis_loss
from src.data.chexpert_labels import CHEXPERT_LABELS


def evaluate(
    model,
    loader: DataLoader,
    tokenizer,
    labeler,
    device: torch.device,
    max_len: int = 128,
    pred_reports: Optional[List[str]] = None,
    pred_label_csv: str | Path | None = None,
    true_label_csv: str | Path | None = None,
    lambda_diagnosis: float = 0.5,
    diag_loss_type: str = "bce",
    semantic_metric: str | None = None,
    bertscore_lang: str = "en",
    bertscore_model: str | None = None,
    bertscore_rescale_with_baseline: bool = True,
    amp_enabled: bool = False,
    amp_dtype: str = "bf16",
    tf32_enabled: bool | None = None,
) -> Dict[str, float]:
    """BLEU/ROUGE와 CheXpert 라벨 정합도를 계산한다.

    인자:
        model: 생성 모델.
        loader (DataLoader): 평가 데이터 로더.
        tokenizer: 토크나이저.
        labeler: 라벨러(또는 None).
        device (torch.device): 디바이스.
        max_len (int): 생성 최대 길이.

    반환:
        Dict[str, float]: 평가 지표 딕셔너리.
    """
    if model is not None:
        model.eval()
    amp_enabled = bool(amp_enabled) and isinstance(device, torch.device) and device.type == "cuda"
    amp_dtype = str(amp_dtype).lower()
    if amp_dtype in {"bf16", "bfloat16"}:
        amp_torch_dtype = torch.bfloat16
    elif amp_dtype in {"fp16", "float16"}:
        amp_torch_dtype = torch.float16
    else:
        raise ValueError(f"Unsupported amp_dtype: {amp_dtype}")

    @contextmanager
    def tf32_context(enabled: bool | None):
        if enabled is None or not torch.cuda.is_available():
            yield
            return
        prev_matmul = torch.backends.cuda.matmul.allow_tf32
        prev_cudnn = torch.backends.cudnn.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = bool(enabled)
        torch.backends.cudnn.allow_tf32 = bool(enabled)
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("high" if enabled else "highest")
        try:
            yield
        finally:
            torch.backends.cuda.matmul.allow_tf32 = prev_matmul
            torch.backends.cudnn.allow_tf32 = prev_cudnn
    refs = []
    hyps = pred_reports or []
    true_labels = []
    total_gen = 0.0
    total_diag = 0.0
    total_loss = 0.0
    steps = 0

    with torch.no_grad():
        for batch in tqdm(loader, desc="eval", leave=False):
            images = batch["image"].to(device)
            impressions = batch.get("impression", [""] * images.shape[0])
            batch_true = batch.get("chex_labels")
            if batch_true is not None:
                true_labels.append(batch_true.detach().cpu().numpy())
            refs.extend(impressions)

            if model is not None:
                if pred_reports is None:
                    for i in range(images.shape[0]):
                        out = model.generate(
                            images[i : i + 1], tokenizer, max_len=max_len, device=device
                        )
                        hyps.append(out["text"])

                # loss 계산(teacher forcing)
                input_ids = batch["input_ids"].to(device)
                labels = batch["labels"].to(device)
                weights = batch["weights"].to(device)
                attention_mask = batch.get("attention_mask")
                if attention_mask is not None:
                    attention_mask = attention_mask.to(device)

                with tf32_context(tf32_enabled):
                    with torch.autocast(
                        device_type=device.type,
                        dtype=amp_torch_dtype,
                        enabled=amp_enabled,
                    ):
                        outputs = model(
                            images=images,
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                        )
                        logits = outputs["logits"]
                        diag_logits = outputs["diag_logits"]

                        gen_loss = weighted_ce_loss(
                            logits[:, -labels.shape[1] :, :], labels, weights
                        )
                        diag_loss_val = torch.tensor(0.0, device=device)
                        if batch_true is not None:
                            diag_loss_val = diagnosis_loss(
                                diag_logits, batch_true.to(device), loss_type=diag_loss_type
                            )
                        loss = gen_loss + lambda_diagnosis * diag_loss_val

                total_gen += gen_loss.item()
                total_diag += diag_loss_val.item()
                total_loss += loss.item()
                steps += 1

    bleu = compute_bleu(refs, hyps)
    rouge = compute_rouge_l(refs, hyps)

    bert_p = bert_r = bert_f1 = None
    if semantic_metric and str(semantic_metric).lower() == "bertscore":
        device_str = str(device) if isinstance(device, torch.device) else device
        bert_p, bert_r, bert_f1 = compute_bertscore(
            refs,
            hyps,
            lang=bertscore_lang,
            model_type=bertscore_model,
            rescale_with_baseline=bertscore_rescale_with_baseline,
            device=device_str,
        )

    pred_labels = None
    if pred_label_csv is not None:
        pred_labels = _load_chex_labels(pred_label_csv)
    elif labeler is not None:
        pred_labels = np.array(labeler(hyps))

    if true_label_csv is not None:
        true_labels = _load_chex_labels(true_label_csv)
    elif true_labels:
        true_labels = np.concatenate(true_labels, axis=0)
    elif labeler is not None:
        true_labels = np.array(labeler(refs))
    else:
        true_labels = None

    if pred_labels is not None and true_labels is not None:
        if pred_labels.shape != true_labels.shape:
            raise ValueError(
                f"pred/true label shape mismatch: {pred_labels.shape} vs {true_labels.shape}"
            )
        acc = (pred_labels == true_labels).mean()
        if diag_loss_type in ("ce3", "ce4"):
            macro_f1, micro_f1 = compute_f1_multiclass(true_labels, pred_labels)
        else:
            macro_f1, micro_f1 = compute_f1(true_labels, pred_labels)
    else:
        acc = 0.0
        macro_f1 = 0.0
        micro_f1 = 0.0

    return {
        "bleu": float(bleu),
        "rouge_l": float(rouge),
        "bertscore_p": None if bert_p is None else float(bert_p),
        "bertscore_r": None if bert_r is None else float(bert_r),
        "bertscore_f1": None if bert_f1 is None else float(bert_f1),
        "chex_acc": float(acc),
        "chex_macro_f1": float(macro_f1),
        "chex_micro_f1": float(micro_f1),
        "eval_loss": total_loss / max(steps, 1) if steps else None,
        "eval_gen_loss": total_gen / max(steps, 1) if steps else None,
        "eval_diag_loss": total_diag / max(steps, 1) if steps else None,
    }


def _load_chex_labels(path: str | Path) -> np.ndarray:
    """CheXpert 라벨 CSV에서 (N,14) 라벨 배열을 로드한다."""
    import pandas as pd

    df = pd.read_csv(path)
    # Reports/Impression 컬럼은 무시하고 라벨만 사용
    missing = [c for c in CHEXPERT_LABELS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing CheXpert label columns in {path}: {missing}")
    labels = df[CHEXPERT_LABELS].fillna(3).astype(float).values
    return labels
