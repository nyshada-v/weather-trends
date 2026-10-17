"""Configuration loading and attribute-style access."""

from __future__ import annotations

import copy
import os
from typing import Any

import yaml

DEFAULT_CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "config.yaml")


class Config(dict):
    """Dictionary with recursive attribute access."""

    def __getattr__(self, item: str) -> Any:
        try:
            value = self[item]
        except KeyError as exc:
            raise AttributeError(item) from exc
        if isinstance(value, dict) and not isinstance(value, Config):
            value = Config(value)
            self[item] = value
        return value

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value


def _merge(base: dict, override: dict) -> dict:
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge(result[key], value)
        else:
            result[key] = value
    return result


def resolve_paths(config: Config, root: str | None = None) -> Config:
    root = root or os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    for section in ("data", "output"):
        if section not in config:
            continue
        for key, value in list(config[section].items()):
            if key.endswith("_dir") and isinstance(value, str):
                config[section][key] = value if os.path.isabs(value) else os.path.join(root, value)
    return config


def load_config(path: str | None = None, overrides: dict | None = None) -> Config:
    path = path or DEFAULT_CONFIG_PATH
    with open(path, "r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if overrides:
        raw = _merge(raw, overrides)
    config = Config(raw)
    resolve_paths(config)
    return config
