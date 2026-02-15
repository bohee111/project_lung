"""마스킹 전/후 라벨 변화율 계산 스크립트."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.chexpert_labels import CHEXPERT_LABELS


def _load_labels(path: Path) -> np.ndarray:
    df = pd.read_csv(path)
    missing = [c for c in CHEXPERT_LABELS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing CheXpert label columns in {path}: {missing}")
    labels = df[CHEXPERT_LABELS].fillna(3).astype(float).values
    return labels


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--orig_labels", required=True, help="마스킹 전 라벨 CSV")
    parser.add_argument("--masked_labels", required=True, help="마스킹 후 라벨 CSV")
    parser.add_argument("--out_csv", default=None, help="변화율 저장 경로(선택)")
    args = parser.parse_args()

    orig = _load_labels(Path(args.orig_labels))
    masked = _load_labels(Path(args.masked_labels))
    if orig.shape != masked.shape:
        raise ValueError(f"shape mismatch: {orig.shape} vs {masked.shape}")

    diff = (orig != masked).astype(float)
    overall = float(diff.mean())
    per_label = diff.mean(axis=0)

    print(f"overall_change_rate: {overall:.4f}")
    for name, val in zip(CHEXPERT_LABELS, per_label):
        print(f"{name}: {val:.4f}")

    if args.out_csv:
        out_path = Path(args.out_csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(
            {"label": CHEXPERT_LABELS, "change_rate": per_label}
        )
        df.loc[len(df.index)] = ["_overall", overall]
        df.to_csv(out_path, index=False)
        print(f"saved to {out_path}")


if __name__ == "__main__":
    main()
