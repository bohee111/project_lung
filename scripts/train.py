"""실데이터 학습 스크립트."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from functools import partial

import torch
import torch.multiprocessing as mp
from torch.utils.data import DataLoader

from src.config import load_config, ensure_dir
from src.data.dataset import ChestXrayDataset, TextConfig, collate_fn
from src.data.tokenizer import get_tokenizer
from src.models.pipeline import LungImpressionModel
from src.training.trainer import Trainer
from src.utils.seed import set_seed


def log_tokenizer_info(cfg: dict, tokenizer) -> None:
    """토크나이저 로딩 상태를 출력한다."""
    tok_cfg = cfg["tokenizer"]
    print("[tokenizer] name_or_path:", tok_cfg.get("name_or_path"))
    print("[tokenizer] local_files_only:", tok_cfg.get("local_files_only"))
    print("[tokenizer] class:", tokenizer.__class__.__name__)
    print("[tokenizer] vocab_size:", len(tokenizer))

    vocab_file = getattr(tokenizer, "vocab_file", None)
    print("[tokenizer] vocab_file:", vocab_file)
    fallback_vocab = Path(__file__).resolve().parents[1] / "assets" / "minibert_vocab.txt"
    print("[tokenizer] fallback_vocab:", fallback_vocab)
    print("[tokenizer] using_fallback_vocab:", str(vocab_file) == str(fallback_vocab))

    special = tok_cfg.get("special_tokens", {})
    for key, token in special.items():
        token_id = tokenizer.convert_tokens_to_ids(token)
        print(f"[tokenizer] special {key}:", token, "id:", token_id)


def main() -> None:
    """설정 파일을 기반으로 학습을 수행하고 체크포인트를 저장한다.

    인자:
        없음

    반환:
        None
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(args.seed)

    mp_start_method = cfg["training"].get("mp_start_method")
    if mp_start_method:
        method = str(mp_start_method).lower()
        if method not in {"none", "null", "false", "off"}:
            current = mp.get_start_method(allow_none=True)
            if current is None:
                mp.set_start_method(method)
                print(f"[mp] start_method set to: {method}")
            elif current != method:
                print(f"[mp] start_method already set to {current}, requested {method}; keeping {current}")

    tf32_enabled = bool(cfg["training"].get("tf32", False))
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = tf32_enabled
        torch.backends.cudnn.allow_tf32 = tf32_enabled
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("high" if tf32_enabled else "highest")
        print(f"[tf32] enabled: {tf32_enabled}")

    tokenizer = get_tokenizer(cfg)
    log_tokenizer_info(cfg, tokenizer)
    tcfg = TextConfig(**cfg["tokenizer"]["special_tokens"])

    dataset = ChestXrayDataset(
        csv_path=cfg["paths"]["data_csv"],
        image_root=cfg["paths"]["image_root"],
        tokenizer=tokenizer,
        text_cfg=tcfg,
        max_length=cfg["tokenizer"]["max_length"],
        chex_csv_path=cfg["paths"].get("chexpert_csv"),
        chex_label_mode=cfg["model"].get("diag_loss_type", "bce"),
        w_find=cfg["training"]["w_find"],
        w_imp=cfg["training"]["w_imp"],
    )

    # 데이터 로더 구성.
    pad_id = tokenizer.pad_token_id
    num_workers = int(cfg["training"].get("num_workers", 0))
    pin_memory = bool(cfg["training"].get("pin_memory", False))
    persistent_workers = bool(cfg["training"].get("persistent_workers", False)) and num_workers > 0
    collate = partial(collate_fn, pad_id=pad_id)
    loader = DataLoader(
        dataset,
        batch_size=cfg["training"]["batch_size"],
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        collate_fn=collate,
    )

    # 평가 데이터 로더 구성(선택).
    eval_loader = None
    eval_csv = cfg["paths"].get("eval_csv")
    if eval_csv and Path(eval_csv).exists():
        eval_image_root = cfg["paths"].get("eval_image_root", cfg["paths"]["image_root"])
        eval_chex_csv = cfg["paths"].get("eval_chexpert_csv")
        eval_dataset = ChestXrayDataset(
            csv_path=eval_csv,
            image_root=eval_image_root,
            tokenizer=tokenizer,
            text_cfg=tcfg,
            max_length=cfg["tokenizer"]["max_length"],
            chex_csv_path=eval_chex_csv,
            chex_label_mode=cfg["model"].get("diag_loss_type", "bce"),
            w_find=cfg["training"]["w_find"],
            w_imp=cfg["training"]["w_imp"],
        )
        eval_loader = DataLoader(
            eval_dataset,
            batch_size=cfg["evaluation"]["batch_size"],
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            collate_fn=collate,
        )

    # 디바이스 결정(auto + CUDA 가용 여부).
    device = torch.device("cuda" if cfg["training"]["device"] == "auto" and torch.cuda.is_available() else "cpu")

    amp_enabled = bool(cfg["training"].get("amp", False))
    amp_dtype = cfg["training"].get("amp_dtype", "bf16")
    if amp_enabled and device.type != "cuda":
        print("[amp] requested but device is cpu; disabling")
        amp_enabled = False
    print(f"[amp] enabled: {amp_enabled}, dtype: {amp_dtype}")

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

    optim_name = str(cfg["training"].get("optimizer", "adamw")).lower()
    if optim_name == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=cfg["training"]["lr"],
            weight_decay=cfg["training"]["weight_decay"],
        )
    elif optim_name == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=cfg["training"]["lr"],
            weight_decay=cfg["training"]["weight_decay"],
        )
    elif optim_name == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=cfg["training"]["lr"],
            weight_decay=cfg["training"]["weight_decay"],
            momentum=float(cfg["training"].get("momentum", 0.9)),
            nesterov=bool(cfg["training"].get("nesterov", True)),
        )
    else:
        raise ValueError(f"Unsupported optimizer: {optim_name}")

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        device=device,
        lambda_diagnosis=cfg["training"]["lambda_diagnosis"],
        diag_loss_type=cfg["model"].get("diag_loss_type", "bce"),
        amp_enabled=amp_enabled,
        amp_dtype=amp_dtype,
    )

    checkpoint_dir = ensure_dir(cfg["paths"]["checkpoint_dir"])
    output_dir = ensure_dir(cfg["paths"]["output_dir"])
    log_path = Path(cfg["training"].get("log_path", output_dir / "train_log.csv"))
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_exists = log_path.exists() and log_path.stat().st_size > 0

    raw_log_every = cfg["training"].get("log_every", 10)
    if isinstance(raw_log_every, bool):
        log_every = 0 if not raw_log_every else 10
    else:
        log_every = int(raw_log_every)

    # 에폭 루프.
    with log_path.open("a", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "epoch",
                "loss",
                "gen_loss",
                "diag_loss",
                "eval_loss",
                "eval_gen_loss",
                "eval_diag_loss",
            ],
        )
        if not log_exists:
            writer.writeheader()

        for epoch in range(1, cfg["training"]["epochs"] + 1):
            metrics = trainer.train_epoch(loader, log_every=log_every)
            eval_metrics = None
            if eval_loader is not None:
                eval_metrics = trainer.eval_epoch(eval_loader)
                print(f"epoch {epoch}: {metrics} | eval: {eval_metrics}")
            else:
                print(f"epoch {epoch}: {metrics}")

            writer.writerow(
                {
                    "epoch": epoch,
                    "loss": metrics["loss"],
                    "gen_loss": metrics["gen_loss"],
                    "diag_loss": metrics["diag_loss"],
                    "eval_loss": None if eval_metrics is None else eval_metrics["loss"],
                    "eval_gen_loss": None if eval_metrics is None else eval_metrics["gen_loss"],
                    "eval_diag_loss": None if eval_metrics is None else eval_metrics["diag_loss"],
                }
            )
            f.flush()

            if epoch % cfg["training"]["save_every"] == 0:
                ckpt_path = checkpoint_dir / f"epoch_{epoch}.pt"
                trainer.save_checkpoint(ckpt_path, epoch)


if __name__ == "__main__":
    main()
