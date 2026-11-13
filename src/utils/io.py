"""Storage helpers.

Tables are written as Parquet (columnar, typed, compressed) when ``pyarrow``
is available, otherwise as CSV. Callers pass a path *without* extension and
:func:`load_table` finds whichever file exists, so the rest of the code never
needs to know the storage format.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def _parquet_available() -> bool:
    """Return True if a Parquet engine (pyarrow or fastparquet) is importable."""
    for engine in ("pyarrow", "fastparquet"):
        try:
            __import__(engine)
            return True
        except ImportError:
            continue
    return False


PARQUET_OK = _parquet_available()


def save_table(df: pd.DataFrame, path_no_ext: Path, fmt: str = "parquet") -> Path:
    """Save a DataFrame as Parquet (preferred) or CSV and return the written path."""
    path_no_ext = Path(path_no_ext)
    path_no_ext.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "parquet" and PARQUET_OK:
        out = path_no_ext.with_suffix(".parquet")
        df.to_parquet(out, index=False)
    else:
        out = path_no_ext.with_suffix(".csv")
        df.to_csv(out, index=False)
    return out


def load_table(path_no_ext: Path, parse_dates: list[str] | None = None) -> pd.DataFrame:
    """Load a table saved by :func:`save_table` (Parquet first, then CSV)."""
    path_no_ext = Path(path_no_ext)
    pq, csv = path_no_ext.with_suffix(".parquet"), path_no_ext.with_suffix(".csv")
    if pq.exists() and PARQUET_OK:
        return pd.read_parquet(pq)
    if csv.exists():
        return pd.read_csv(csv, parse_dates=parse_dates or [])
    raise FileNotFoundError(f"No table found at {pq} or {csv}. Run the upstream pipeline stage first.")


def save_json(obj: Any, path: Path) -> Path:
    """Write a JSON file (pretty-printed; non-serialisable values converted to str)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, default=str)
    return path


def load_json(path: Path) -> Any:
    """Read a JSON file."""
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)
