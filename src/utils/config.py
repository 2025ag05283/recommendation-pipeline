"""Configuration loading.

All pipeline stages call :func:`load_config`, which reads ``configs/config.yaml``
and resolves every entry under ``paths`` to an absolute :class:`pathlib.Path`
anchored at the repository root. Using one config object everywhere is what
makes the pipeline reproducible.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import yaml

# Repository root = two levels above this file (src/utils/config.py -> repo root).
ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs" / "config.yaml"


def load_config(path: Optional[str | Path] = None) -> Dict[str, Any]:
    """Load the YAML config and return it as a dict with resolved, existing paths.

    Args:
        path: optional path to an alternative config file.

    Returns:
        Config dictionary. ``cfg["paths"][name]`` is an absolute Path and
        ``cfg["root"]`` is the repository root.
    """
    cfg_path = Path(path) if path else DEFAULT_CONFIG
    with open(cfg_path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)

    cfg["root"] = ROOT
    cfg["config_path"] = cfg_path
    # Resolve every relative path against the repo root and make sure the folder exists.
    for key, rel in cfg["paths"].items():
        p = (ROOT / rel).resolve()
        p.mkdir(parents=True, exist_ok=True)
        cfg["paths"][key] = p
    return cfg
