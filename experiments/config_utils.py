from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Tuple

import yaml


def load_yaml_config(config_path: str) -> Dict[str, Any]:
    if not config_path:
        return {}
    p = Path(config_path)
    if not p.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    with p.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config root must be a mapping: {config_path}")
    return data


def _parse_value(raw: str) -> Any:
    try:
        return yaml.safe_load(raw)
    except Exception:
        return raw


def parse_unknown_overrides(unknown: List[str]) -> Dict[str, Any]:
    """Parse unknown argparse tokens as key/value overrides.

    Supports:
      --foo 123
      --bar true
      --flag
    """
    out: Dict[str, Any] = {}
    i = 0
    while i < len(unknown):
        tok = unknown[i]
        if not tok.startswith("--"):
            i += 1
            continue
        key = tok[2:].replace("-", "_")
        if i + 1 < len(unknown) and not unknown[i + 1].startswith("--"):
            out[key] = _parse_value(unknown[i + 1])
            i += 2
        else:
            out[key] = True
            i += 1
    return out


def apply_overrides(namespace: argparse.Namespace, overrides: Dict[str, Any]) -> argparse.Namespace:
    data = vars(namespace).copy()
    data.update(overrides)
    return argparse.Namespace(**data)


def parse_with_config(parser: argparse.ArgumentParser) -> Tuple[argparse.Namespace, Dict[str, Any]]:
    """Two-pass parse: load --config first, then parse full args + unknown overrides."""
    base = argparse.ArgumentParser(add_help=False)
    base.add_argument("--config", type=str, default="")
    cfg_args, remaining = base.parse_known_args()
    cfg = load_yaml_config(cfg_args.config)

    if cfg:
        parser.set_defaults(**cfg)
    args, unknown = parser.parse_known_args(remaining)
    args.config = cfg_args.config
    overrides = parse_unknown_overrides(unknown)
    if overrides:
        args = apply_overrides(args, overrides)
    return args, cfg

