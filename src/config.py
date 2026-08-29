"""Shared config-loading infrastructure. Not a component -- like contracts.py, every leaf
component is allowed to import this (it holds no cross-component types or logic), so it doesn't
violate the "components never import each other" rule.
"""

from __future__ import annotations

import functools
from pathlib import Path

import yaml

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.yaml"


@functools.lru_cache(maxsize=8)
def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> dict:
    with Path(path).open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)
