"""체크포인트 로드 후 단일 이미지 추론 스크립트."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config, ensure_dir
from src.data.dataset import build_image_transform, load_image
from src.data.tokenizer import get_tokenizer
from src.models.pipeline import LungImpressionModel


def save_mask(out_dir: Path, masks: torch.Tensor) -> None:
    """(B,3,H,W) 마스크를 클래스 맵으로 저장."""
    cls = torch.zeros_like(masks[0, 0], dtype=torch.uint8)
    cls[masks[0, 0] > 0] = 1
    cls[masks[0, 1] > 0] = 2
    cls[masks[0, 2] > 0] = 3
    Image.fromarray(cls.cpu().numpy()).save(out_dir / "mask_classes.png")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="config.yaml 경로")
    parser.add_argument("--checkpoint", required=True, help="체크포인트 경로")
    parser.add_argument("--image", required=True, help="입력 이미지 경로")
    parser.add_argument("--device", default="auto", help="cuda/cpu/auto")
    parser.add_argument("--max_len", type=int, default=None, help="생성 최대 길이")
    parser.add_argument("--out_dir", default=None, help="결과 저장 경로")
    parser.add_argument("--save_mask", action="store_true", help="마스크 저장 여부")
    parser.add_argument("--zero_prefix", action="store_true", help="이미지 프리픽스를 0으로 두고 생성(조건부 여부 확인)")
    parser.add_argument("--show_special", action="store_true", help="special token 포함 출력")
    args = parser.parse_args()

    cfg = load_config(args.config)
    tokenizer = get_tokenizer(cfg)

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = LungImpressionModel(
        vocab_size=len(tokenizer),
        d_model=cfg["model"]["d_model"],
        n_heads=cfg["model"]["n_heads"],
        n_layers=cfg["model"]["n_layers"],
        dropout=cfg["model"]["dropout"],
        diag_classes=cfg["model"]["diag_classes"],
        n_visual_tokens=cfg["model"]["n_visual_tokens"],
        n_diag_tokens=cfg["model"]["n_diag_tokens"],
        decoder_type=cfg["model"].get("decoder_type", "custom"),
        decoder_name_or_path=cfg["model"].get("decoder_name_or_path"),
        decoder_local_files_only=cfg["model"].get("decoder_local_files_only", False),
        decoder_trust_remote_code=cfg["model"].get("decoder_trust_remote_code", False),
        decoder_revision=cfg["model"].get("decoder_revision"),
        visual_encoder=cfg["model"].get("visual_encoder", "resnet50"),
        visual_encoder_backend=cfg["model"].get("visual_encoder_backend", "timm"),
        visual_pretrained=cfg["model"]["visual_pretrained"],
        freeze_encoder=cfg["model"].get("freeze_encoder", False),
        diag_loss_type=cfg["model"].get("diag_loss_type", "bce"),
        segmentation_cfg=cfg.get("segmentation"),
    ).to(device)

    ckpt = torch.load(args.checkpoint, map_location=device)
    state_dict = ckpt.get("state_dict") if isinstance(ckpt, dict) else ckpt
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    image = build_image_transform()(load_image(Path(args.image)))
    images = image.unsqueeze(0).to(device)

    eval_cfg = cfg.get("evaluation", {})
    max_len = args.max_len if args.max_len is not None else eval_cfg.get("max_gen_len", 128)
    prompt_mode = eval_cfg.get("prompt_mode", "impression_only")
    gen_kwargs = {
        "max_len": int(max_len),
        "min_len": int(eval_cfg.get("min_gen_len", 0)),
        "do_sample": bool(eval_cfg.get("do_sample", False)),
        "temperature": float(eval_cfg.get("temperature", 1.0)),
        "top_k": int(eval_cfg.get("top_k", 0)),
        "top_p": float(eval_cfg.get("top_p", 1.0)),
        "repetition_penalty": float(eval_cfg.get("repetition_penalty", 1.0)),
        "no_repeat_ngram_size": int(eval_cfg.get("no_repeat_ngram_size", 0)),
        "stop_on_eos": bool(eval_cfg.get("stop_on_eos", True)),
        "prompt_mode": prompt_mode,
        "max_findings_len": int(eval_cfg.get("max_findings_len", 0)),
        "impression_bias": float(eval_cfg.get("impression_bias", 0.0)),
        "impression_bias_start": eval_cfg.get("impression_bias_start"),
        "zero_prefix": bool(args.zero_prefix),
    }
    with torch.inference_mode():
        out = model.generate(images, tokenizer, device=device, **gen_kwargs)

    text = out["text_with_special"] if args.show_special else out["text"]
    print("origin")
    print(text)
    first_findings = str(prompt_mode or "").lower().replace(" ", "_") in {
        "first_findings",
        "first_finding",
        "findings_first",
    }
    if first_findings and bool(out.get("has_impression", False)):
        text = out.get("impression_text", text)
    print(text)

    if args.save_mask or args.out_dir:
        out_dir = Path(args.out_dir) if args.out_dir else ensure_dir(cfg["paths"]["output_dir"])
        out_dir.mkdir(parents=True, exist_ok=True)
        if args.save_mask:
            save_mask(out_dir, out["masks"])


if __name__ == "__main__":
    main()
