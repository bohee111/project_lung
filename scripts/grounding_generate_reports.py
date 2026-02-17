"""어텐션 기반 마스킹 전/후 리포트 생성 스크립트."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

import torch
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config
from src.data.dataset import load_image, build_image_transform
from src.data.tokenizer import get_tokenizer
from src.evaluation.attention import attention_to_heatmap, save_attention_overlay
from src.evaluation.grounding import mask_topk
from src.models.pipeline import LungImpressionModel


def _collect_images(image_dir: Path) -> list[Path]:
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".dcm", ".dicom"}
    paths = [p for p in image_dir.rglob("*") if p.suffix.lower() in exts and p.is_file()]
    return sorted(paths, key=lambda p: p.as_posix())


def _topk_mask_from_heatmap(heatmap: torch.Tensor | None, k: float) -> torch.Tensor | None:
    if heatmap is None:
        return None
    flat = heatmap.flatten()
    thresh = torch.quantile(flat, 1 - k)
    return heatmap >= thresh


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--image_dir", default=None, help="평가 이미지 폴더 (기본: config의 eval_image_root)")
    parser.add_argument("--out_orig_csv", default="outputs/grounding_original.csv")
    parser.add_argument("--out_masked_csv", default="outputs/grounding_masked.csv")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    cfg = load_config(args.config)
    tokenizer = get_tokenizer(cfg)

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    eval_cfg = cfg.get("evaluation", {})
    grounding_cfg = cfg.get("grounding", {})
    prompt_mode = eval_cfg.get("prompt_mode", "impression_only")
    gen_kwargs = {
        "max_len": int(eval_cfg.get("max_gen_len", 128)),
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
    }

    out_orig = Path(grounding_cfg.get("out_orig_csv", args.out_orig_csv))
    out_masked = Path(grounding_cfg.get("out_masked_csv", args.out_masked_csv))
    overlay_dir = grounding_cfg.get("overlay_dir")
    save_overlay = bool(grounding_cfg.get("save_overlay", False))
    overlay_mode = str(grounding_cfg.get("overlay_mode", "masked")).lower()
    overlay_dir = Path(overlay_dir) if overlay_dir else None

    image_dir = Path(
        grounding_cfg.get("image_dir")
        or args.image_dir
        or cfg["paths"].get("eval_image_root")
        or cfg["paths"]["image_root"]
    )
    image_paths = _collect_images(image_dir)
    if not image_paths:
        raise ValueError(f"No images found in {image_dir}")

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

    ckpt = torch.load(args.ckpt, map_location=device)
    state_dict = ckpt.get("state_dict") if isinstance(ckpt, dict) else ckpt
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    k = float(grounding_cfg.get("top_k_mask", eval_cfg.get("top_k_mask", 0.1)))

    out_orig.parent.mkdir(parents=True, exist_ok=True)
    out_masked.parent.mkdir(parents=True, exist_ok=True)
    if save_overlay and overlay_dir is not None:
        overlay_dir.mkdir(parents=True, exist_ok=True)

    transform = build_image_transform()

    first_findings = str(prompt_mode or "").lower().replace(" ", "_") in {
        "first_findings",
        "first_finding",
        "findings_first",
    }

    with out_orig.open("w", newline="", encoding="utf-8") as f_orig, out_masked.open(
        "w", newline="", encoding="utf-8"
    ) as f_masked:
        w_orig = csv.writer(f_orig, quoting=csv.QUOTE_ALL)
        w_masked = csv.writer(f_masked, quoting=csv.QUOTE_ALL)

        with torch.inference_mode():
            for path in tqdm(image_paths, desc="grounding-generate", unit="img"):
                image = transform(load_image(path))
                out = model.generate(image.unsqueeze(0).to(device), tokenizer, device=device, **gen_kwargs)
                heatmap = attention_to_heatmap(out["attn_tokens"], out["masks"])
                if save_overlay and overlay_dir is not None:
                    if overlay_mode == "attention":
                        overlay_path = overlay_dir / f"{path.stem}_overlay_attn.png"
                        save_attention_overlay(image, heatmap, overlay_path)
                    elif overlay_mode == "both":
                        overlay_path = overlay_dir / f"{path.stem}_overlay_attn.png"
                        save_attention_overlay(image, heatmap, overlay_path)
                        mask = _topk_mask_from_heatmap(heatmap, k=k)
                        mask_heatmap = mask.float() if mask is not None else None
                        overlay_path = overlay_dir / f"{path.stem}_overlay_mask.png"
                        save_attention_overlay(image, mask_heatmap, overlay_path)
                    else:
                        mask = _topk_mask_from_heatmap(heatmap, k=k)
                        mask_heatmap = mask.float() if mask is not None else None
                        overlay_path = overlay_dir / f"{path.stem}_overlay_mask.png"
                        save_attention_overlay(image, mask_heatmap, overlay_path)
                masked = mask_topk(image, heatmap, k=k)
                masked_out = model.generate(
                    masked.unsqueeze(0).to(device), tokenizer, device=device, **gen_kwargs
                )

                orig_text = out["text"]
                masked_text = masked_out["text"]
                if first_findings:
                    if bool(out.get("has_impression", False)):
                        orig_text = out.get("impression_text", orig_text)
                    if bool(masked_out.get("has_impression", False)):
                        masked_text = masked_out.get("impression_text", masked_text)

                w_orig.writerow([orig_text.replace("\n", " ").strip()])
                w_masked.writerow([masked_text.replace("\n", " ").strip()])

    print(f"saved original reports to {out_orig}")
    print(f"saved masked reports to {out_masked}")


if __name__ == "__main__":
    main()
