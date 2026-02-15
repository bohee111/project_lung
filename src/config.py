"""YAML 설정 로딩 및 경로 유틸."""

from __future__ import annotations

from pathlib import Path
import yaml


def load_config(path: str | Path) -> dict:
    """YAML 설정 파일을 읽어 dict로 반환한다.

    인자:
        path (str | Path): 설정 파일 경로.

    반환:
        dict: 파싱된 설정.
    """
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg


def ensure_dir(path: str | Path) -> Path:
    """디렉터리가 없으면 생성하고 Path를 반환한다.

    인자:
        path (str | Path): 생성할 디렉터리 경로.

    반환:
        Path: 생성/확인된 디렉터리 경로.
    """
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p
