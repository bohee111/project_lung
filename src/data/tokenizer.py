"""토크나이저 로딩 및 특수 토큰 설정 유틸."""

from __future__ import annotations

from pathlib import Path
from transformers import BertTokenizerFast


def get_tokenizer(cfg: dict):
    """설정에 따라 BERT 토크나이저를 로드하고 특수 토큰을 등록한다.

    인자:
        cfg (dict): 전체 설정 딕셔너리.

    반환:
        BertTokenizerFast: 구성된 토크나이저.
    """
    tok_cfg = cfg["tokenizer"]
    name_or_path = tok_cfg.get("name_or_path", "bert-base-uncased")
    local_only = tok_cfg.get("local_files_only", True)
    try:
        tokenizer = BertTokenizerFast.from_pretrained(
            name_or_path,
            local_files_only=local_only,
        )
    except Exception:
        # 로컬 파일만 허용되므로 실패 시 최소 어휘로 fallback.
        vocab_file = Path(__file__).resolve().parents[2] / "assets" / "minibert_vocab.txt"
        tokenizer = BertTokenizerFast(vocab_file=str(vocab_file))

    special = tok_cfg["special_tokens"]
    tokenizer.add_special_tokens(
        {
            "bos_token": special["bos"],
            "eos_token": special["eos"],
            "pad_token": special["pad"],
            "additional_special_tokens": [special["findings"], special["impression"]],
        }
    )
    return tokenizer
