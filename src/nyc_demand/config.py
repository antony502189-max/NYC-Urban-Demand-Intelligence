from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML configuration file relative to the project root when needed."""
    config_path = Path(path)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as stream:
        payload = yaml.safe_load(stream)

    if not isinstance(payload, dict):
        raise ValueError(f"Expected a YAML mapping in {config_path}")

    return payload


def ensure_directory(path: str | Path) -> Path:
    """Create a directory if needed and return its absolute path."""
    directory = Path(path)
    if not directory.is_absolute():
        directory = PROJECT_ROOT / directory
    directory.mkdir(parents=True, exist_ok=True)
    return directory
