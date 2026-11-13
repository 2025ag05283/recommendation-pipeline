"""Rule-based data-quality checks.

Every check is a small function returning a boolean Series (``True`` = row
FAILS the rule). :func:`apply_checks` runs a list of checks, records every
failed rule per row, and splits the table into *valid* rows and an *error
dataset* (original values + an ``error_reasons`` column).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple

import pandas as pd


@dataclass
class Check:
    """A named validation rule applied to one table."""
    name: str                                   # short id used in reports, e.g. "rating_in_range"
    category: str                               # missing | type | range | duplicate | timestamp | referential
    description: str                            # human-readable rule
    fails: Callable[[pd.DataFrame], pd.Series]  # returns True where the row violates the rule


# ------------------------------------------------------------------ helpers
def is_blank(s: pd.Series) -> pd.Series:
    """True where a value is missing or an empty / whitespace-only string."""
    return s.isna() | s.astype("string").str.strip().eq("").fillna(True)


def not_integer(s: pd.Series) -> pd.Series:
    """True where a *present* value cannot be parsed as an integer (e.g. 'abc', '3.5')."""
    num = pd.to_numeric(s, errors="coerce")
    present = ~is_blank(s)
    return present & (num.isna() | (num % 1 != 0))


def as_number(s: pd.Series) -> pd.Series:
    """Parse to float (invalid -> NaN)."""
    return pd.to_numeric(s, errors="coerce")


# ---------------------------------------------------------------- rule sets
def user_checks(vcfg: Dict) -> List[Check]:
    """Rules for the users table."""
    lo, hi, genders = vcfg["min_age"], vcfg["max_age"], set(vcfg["allowed_genders"])
    return [
        Check("user_required_fields", "missing", "user_id, age and gender must be present",
              lambda d: is_blank(d["user_id"]) | is_blank(d["age"]) | is_blank(d["gender"])),
        Check("user_id_integer", "type", "user_id must be an integer", lambda d: not_integer(d["user_id"])),
        Check("age_integer", "type", "age must be an integer", lambda d: not_integer(d["age"])),
        Check("age_in_range", "range", f"age must be within [{lo}, {hi}]",
              lambda d: as_number(d["age"]).notna() & ~as_number(d["age"]).between(lo, hi)),
        Check("gender_allowed", "range", f"gender must be one of {sorted(genders)}",
              lambda d: ~is_blank(d["gender"]) & ~d["gender"].astype("string").str.strip().str.upper().isin(genders).fillna(False)),
        Check("user_id_unique", "duplicate", "user_id must be unique (first occurrence kept)",
              lambda d: d["user_id"].duplicated(keep="first") & ~is_blank(d["user_id"])),
    ]


def item_checks(genre_cols: List[str]) -> List[Check]:
    """Rules for the items table."""
    def bad_date(d: pd.DataFrame) -> pd.Series:
        present = ~is_blank(d["release_date"])
        parsed = pd.to_datetime(d["release_date"], format="%d-%b-%Y", errors="coerce")
        return present & parsed.isna()

    def bad_genre(d: pd.DataFrame) -> pd.Series:
        flags = d[genre_cols].apply(pd.to_numeric, errors="coerce")
        return ~flags.isin([0, 1]).all(axis=1)

    return [
        Check("item_required_fields", "missing", "item_id and title must be present",
              lambda d: is_blank(d["item_id"]) | is_blank(d["title"])),
        Check("item_id_integer", "type", "item_id must be an integer", lambda d: not_integer(d["item_id"])),
        Check("release_date_format", "timestamp", "release_date, if present, must be a valid DD-Mon-YYYY date", bad_date),
        Check("genre_flags_binary", "type", "every genre flag must be 0 or 1", bad_genre),
        Check("item_id_unique", "duplicate", "item_id must be unique (first occurrence kept)",
              lambda d: d["item_id"].duplicated(keep="first") & ~is_blank(d["item_id"])),
    ]


def interaction_checks(vcfg: Dict, valid_user_ids: set, valid_item_ids: set) -> List[Check]:
    """Rules for the interactions table (needs the valid user / item ids for referential integrity)."""
    rmin, rmax = vcfg["rating_min"], vcfg["rating_max"]
    min_ts = pd.Timestamp(vcfg["min_timestamp"]).value // 10**9
    now_ts = pd.Timestamp.now().value // 10**9
    return [
        Check("interaction_required_fields", "missing", "user_id, item_id, rating and timestamp must be present",
              lambda d: d[["user_id", "item_id", "rating", "timestamp"]].apply(is_blank).any(axis=1)),
        Check("ids_integer", "type", "user_id and item_id must be integers",
              lambda d: not_integer(d["user_id"]) | not_integer(d["item_id"])),
        Check("rating_numeric", "type", "rating must be numeric", lambda d: ~is_blank(d["rating"]) & as_number(d["rating"]).isna()),
        Check("rating_valid", "range", f"rating must be an integer in [{rmin}, {rmax}]",
              lambda d: as_number(d["rating"]).notna() & (~as_number(d["rating"]).between(rmin, rmax) | (as_number(d["rating"]) % 1 != 0))),
        Check("timestamp_numeric", "timestamp", "timestamp must be a unix epoch integer",
              lambda d: not_integer(d["timestamp"])),
        Check("timestamp_plausible", "timestamp", f"timestamp must be between {vcfg['min_timestamp']} and now",
              lambda d: as_number(d["timestamp"]).notna() & ~as_number(d["timestamp"]).between(min_ts, now_ts)),
        Check("no_exact_duplicates", "duplicate", "exact duplicate interaction rows are not allowed (first kept)",
              lambda d: d.duplicated(subset=["user_id", "item_id", "rating", "timestamp"], keep="first")),
        Check("user_exists", "referential", "user_id must exist in the validated users table",
              lambda d: ~is_blank(d["user_id"]) & ~as_number(d["user_id"]).isin(valid_user_ids)),
        Check("item_exists", "referential", "item_id must exist in the validated items table",
              lambda d: ~is_blank(d["item_id"]) & ~as_number(d["item_id"]).isin(valid_item_ids)),
    ]


# ---------------------------------------------------------------- execution
def apply_checks(df: pd.DataFrame, checks: List[Check], table: str) -> Tuple[pd.DataFrame, pd.DataFrame, List[Dict]]:
    """Run all checks on ``df``.

    Returns:
        valid rows, error rows (with ``error_reasons`` and ``table`` columns),
        and a per-check result list (name, category, description, failed count, status).
    """
    df = df.reset_index(drop=True)
    reasons = pd.Series([[] for _ in range(len(df))], index=df.index, dtype=object)
    results = []
    for chk in checks:
        mask = chk.fails(df).fillna(True).astype(bool)        # a rule that cannot be evaluated counts as failed
        for i in df.index[mask]:
            reasons.at[i].append(chk.name)
        results.append({"table": table, "check": chk.name, "category": chk.category,
                        "rule": chk.description, "failed_rows": int(mask.sum()),
                        "status": "PASS" if mask.sum() == 0 else "FAIL"})
    bad = reasons.map(len) > 0
    errors = df[bad].copy()
    errors["error_reasons"] = reasons[bad].map(";".join)
    errors.insert(0, "table", table)
    return df[~bad].copy(), errors, results
