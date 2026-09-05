from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import yaml

DEFAULT_CONFIG_PATH = Path("config/config.yaml")


@lru_cache(maxsize=1)
def load_config() -> dict:
    path = Path(os.environ.get("MLOPS_CONFIG", str(DEFAULT_CONFIG_PATH)))
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def get(*keys: str, default=None):
    value = load_config()
    for key in keys:
        value = value.get(key) if isinstance(value, dict) else None
        if value is None:
            return default
    return value
