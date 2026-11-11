from __future__ import annotations

from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = ROOT / "configs" / "default.yaml"


def load_config(path=DEFAULT_CONFIG):
    path = Path(path).resolve()
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Configuration must be a mapping")
    if not isinstance(raw.get("datasets"), dict) or not raw["datasets"]:
        raise ValueError("Configuration must contain a nonempty datasets mapping")
    if not isinstance(raw.get("formulas"), dict) or not raw["formulas"]:
        raise ValueError("Configuration must contain a nonempty formulas mapping")
    return raw, path
