"""Configuration loading. All scripts call `load_config()`; paths are resolved relative to the project root."""
from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]


def _set_nested(cfg: dict, dotted: str, value: Any) -> None:
    keys = dotted.split(".")
    node = cfg
    for k in keys[:-1]:
        node = node.setdefault(k, {})
    node[keys[-1]] = value


def _merge(base: dict, extra: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in extra.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path: str | Path | None = None, overrides: list[str] | None = None) -> dict:
    """Load configs/default.yaml, then merge `path` (if it differs), then apply `key.sub=value` overrides."""
    default = ROOT / "configs" / "default.yaml"
    cfg = yaml.safe_load(default.read_text())
    if path is not None and Path(path).resolve() != default.resolve():
        cfg = _merge(cfg, yaml.safe_load(Path(path).read_text()) or {})
    for ov in overrides or []:
        key, _, raw = ov.partition("=")
        _set_nested(cfg, key.strip(), yaml.safe_load(raw))
    for k, v in cfg["paths"].items():
        p = Path(v)
        cfg["paths"][k] = str(p if p.is_absolute() else ROOT / p)
    return cfg


def cli(description: str, extra_args=None) -> tuple[dict, argparse.Namespace]:
    """Standard command-line interface shared by every script."""
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"))
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                    help="override config values, e.g. --set models.seeds=[0] explain.lime_samples=50")
    if extra_args:
        extra_args(ap)
    args = ap.parse_args()
    return load_config(args.config, args.set), args
