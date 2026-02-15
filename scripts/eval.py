"""학습된 체크포인트 평가 스크립트."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config
from src.data.dataset import ChestXrayDataset, TextConfig, collate_fn
from src.data.tokenizer import get_tokenizer
from src.models.pipeline import LungImpressionModel
from src.evaluation.evaluate import evaluate


def main() -> None:
    """체크포인트를 로드해 평가 지표를 출력한다.

    인자:
        없음

    반환:
        None
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--ckpt", default=None, help="loss 계산 시 사용할 체크포인트")
    parser.add_argument("--with_loss", action="store_true", help="체크포인트 로드 후 eval loss 계산")
    parser.add_argument("--amp", action="store_true", help="eval loss 계산 시 AMP 사용")
    parser.add_argument("--no-amp", action="store_true", help="eval loss 계산 시 AMP 비사용")
    parser.add_argument("--amp-dtype", default=None, help="AMP dtype (bf16|fp16)")
    parser.add_argument("--tf32", action="store_true", help="eval loss 계산 시 TF32 사용")
    parser.add_argument("--no-tf32", action="store_true", help="eval loss 계산 시 TF32 비사용")
    args = parser.parse_args()

    cfg = load_config(args.config)
    tokenizer = get_tokenizer(cfg)
    tcfg = TextConfig(**cfg["tokenizer"]["special_tokens"])

    eval_csv = cfg["paths"].get("eval_csv", cfg["paths"]["data_csv"])
    eval_chex_csv = cfg["paths"].get("eval_chexpert_csv")
    eval_image_root = cfg["paths"].get("eval_image_root", cfg["paths"]["image_root"])
    dataset = ChestXrayDataset(
        csv_path=eval_csv,
        image_root=eval_image_root,
        tokenizer=tokenizer,
        text_cfg=tcfg,
        max_length=cfg["tokenizer"]["max_length"],
        chex_csv_path=eval_chex_csv,
        chex_label_mode=cfg["model"].get("diag_loss_type", "bce"),
        w_find=cfg["training"].get("w_find", 1.0),
        w_imp=cfg["training"].get("w_imp", 3.0),
    )

    # 평가 데이터 로더.
    pad_id = tokenizer.pad_token_id
    loader = DataLoader(
        dataset,
        batch_size=cfg["evaluation"]["batch_size"],
        shuffle=False,
        num_workers=cfg["training"]["num_workers"],
        collate_fn=lambda b: collate_fn(b, pad_id),
    )

    device = torch.device("cuda" if cfg["training"]["device"] == "auto" and torch.cuda.is_available() else "cpu")

    model = None
    if args.with_loss:
        if not args.ckpt:
            raise ValueError("--with_loss 옵션에는 --ckpt가 필요합니다.")
        model = LungImpressionModel(
            vocab_size=len(tokenizer),
            d_model=cfg["model"]["d_model"],
            n_heads=cfg["model"]["n_heads"],
            n_layers=cfg["model"]["n_layers"],
            dropout=cfg["model"]["dropout"],
            diag_classes=cfg["model"]["diag_classes"],
        n_visual_tokens=cfg["model"]["n_visual_tokens"],
        n_diag_tokens=cfg["model"]["n_diag_tokens"],
        visual_encoder=cfg["model"].get("visual_encoder", "resnet50"),
        visual_encoder_backend=cfg["model"].get("visual_encoder_backend", "timm"),
        visual_pretrained=cfg["model"]["visual_pretrained"],
        freeze_encoder=cfg["model"].get("freeze_encoder", False),
        diag_loss_type=cfg["model"].get("diag_loss_type", "bce"),
        segmentation_cfg=cfg.get("segmentation"),
    ).to(device)

        ckpt = torch.load(args.ckpt, map_location=device)
        state_dict = ckpt.get("state_dict") if isinstance(ckpt, dict) else ckpt
        model.load_state_dict(state_dict)

    labeler = None

    pred_reports = None
    pred_reports_path = cfg["paths"].get("eval_pred_reports_csv")
    if pred_reports_path and Path(pred_reports_path).exists():
        import csv as _csv

        pred_reports = []
        with Path(pred_reports_path).open("r", encoding="utf-8", newline="") as f:
            reader = _csv.reader(f)
            for row in reader:
                if not row:
                    pred_reports.append("")
                else:
                    pred_reports.append(row[0])

    pred_label_csv = cfg["paths"].get("eval_pred_chexpert_csv")

    eval_cfg = cfg.get("evaluation", {})
    amp_enabled = bool(cfg["training"].get("amp", False))
    if args.amp:
        amp_enabled = True
    if args.no_amp:
        amp_enabled = False
    amp_dtype = args.amp_dtype or cfg["training"].get("amp_dtype", "bf16")

    tf32_enabled = bool(cfg["training"].get("tf32", False))
    if args.tf32:
        tf32_enabled = True
    if args.no_tf32:
        tf32_enabled = False
    metrics = evaluate(
        model=model,
        loader=loader,
        tokenizer=tokenizer,
        labeler=labeler,
        device=device,
        max_len=eval_cfg.get("max_gen_len", 128),
        pred_reports=pred_reports,
        pred_label_csv=pred_label_csv,
        true_label_csv=eval_chex_csv,
        diag_loss_type=cfg["model"].get("diag_loss_type", "bce"),
        semantic_metric=eval_cfg.get("semantic_metric"),
        bertscore_lang=eval_cfg.get("bertscore_lang", "en"),
        bertscore_model=eval_cfg.get("bertscore_model"),
        bertscore_rescale_with_baseline=bool(
            eval_cfg.get("bertscore_rescale_with_baseline", True)
        ),
        amp_enabled=amp_enabled,
        amp_dtype=amp_dtype,
        tf32_enabled=tf32_enabled,
    )

    print(metrics)


if __name__ == "__main__":
    main()
