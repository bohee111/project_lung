"""체크포인트로 eval CSV 전체 추론 후 결과를 CSV로 저장."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys
from functools import partial

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config, ensure_dir
from src.data.dataset import ChestXrayDataset, TextConfig, collate_fn
from src.data.tokenizer import get_tokenizer
from src.models.pipeline import LungImpressionModel


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out_csv", default=None, help="예측 문장 저장 경로(헤더 없음)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    eval_csv = cfg["paths"].get("eval_csv", cfg["paths"]["data_csv"])
    out_csv = args.out_csv or cfg["paths"].get("eval_pred_reports_csv") or "outputs/eval_pred_reports.csv"

    tokenizer = get_tokenizer(cfg)
    tcfg = TextConfig(**cfg["tokenizer"]["special_tokens"])

    eval_image_root = cfg["paths"].get("eval_image_root", cfg["paths"]["image_root"])
    dataset = ChestXrayDataset(
        csv_path=eval_csv,
        image_root=eval_image_root,
        tokenizer=tokenizer,
        text_cfg=tcfg,
        max_length=cfg["tokenizer"]["max_length"],
    )

    pad_id = tokenizer.pad_token_id
    collate = partial(collate_fn, pad_id=pad_id)
    loader = DataLoader(
        dataset,
        batch_size=cfg["evaluation"]["batch_size"],
        shuffle=False,
        num_workers=cfg["training"]["num_workers"],
        collate_fn=collate,
    )

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

    ckpt = torch.load(args.ckpt, map_location=device)
    state_dict = ckpt.get("state_dict") if isinstance(ckpt, dict) else ckpt
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    eval_cfg = cfg.get("evaluation", {})
    max_len = eval_cfg.get("max_gen_len", 128)
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
    }
    out_path = Path(out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_ALL)
        first_findings = str(prompt_mode or "").lower().replace(" ", "_") in {
            "first_findings",
            "first_finding",
            "findings_first",
        }
        sample_idx = 0
        with torch.inference_mode():
            pbar = tqdm(total=len(dataset), desc="generate", unit="img")
            for batch in loader:
                images = batch["image"].to(device)
                for i in range(images.shape[0]):
                    out = model.generate(images[i : i + 1], tokenizer, device=device, **gen_kwargs)
                    text = out["text"]
                    if first_findings:
                        has_imp = bool(out.get("has_impression", False))
                        print(f"[impression_token] idx={sample_idx} has_impression={has_imp}")
                        if has_imp:
                            text = out.get("impression_text", text)
                    text = text.replace("\n", " ").strip()
                    writer.writerow([text])
                    sample_idx += 1
                pbar.update(images.shape[0])
            pbar.close()

    print(f"saved predictions to {out_path}")


if __name__ == "__main__":
    main()
