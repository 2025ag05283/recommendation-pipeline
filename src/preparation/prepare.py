"""Stage 3 - Data preparation.

Takes the *validated* tables and produces clean, analysis-ready data:

* **Users** - standardise gender (upper-case) and occupation (lower-case, trimmed);
  impute missing ``occupation`` / ``zip_code`` with ``"unknown"``; add an age group.
* **Items** - parse ``release_date``; impute a missing date with the median
  release date (flag ``release_date_imputed``); derive ``release_year``; build a
  pipe-separated ``genres`` string and ``n_genres``; drop the always-empty
  ``video_release_date`` column.
* **Interactions** - convert unix time to datetime; if the same user rated the
  same item more than once keep the most recent rating (logged).
* **Time-based split** - all interactions up to the 80th percentile timestamp
  are TRAIN, later ones are TEST. This mimics production (train on the past,
  predict the future) and prevents look-ahead leakage.
"""
from __future__ import annotations

from typing import Dict

import pandas as pd

from src.ingestion.ingest import GENRES
from src.utils.io import load_table, save_json, save_table
from src.utils.logger import get_logger

log = get_logger(__name__)
GENRE_COLS = [f"genre_{g}" for g in GENRES]


def clean_users(users: pd.DataFrame) -> pd.DataFrame:
    """Standardise formats and impute optional user fields."""
    u = users.copy()
    u["gender"] = u["gender"].astype("string").str.strip().str.upper()
    u["occupation"] = u["occupation"].astype("string").str.strip().str.lower().fillna("unknown").replace("", "unknown")
    u["zip_code"] = u["zip_code"].astype("string").str.strip().fillna("unknown").replace("", "unknown")
    u["age_group"] = pd.cut(u["age"], bins=[0, 17, 24, 34, 44, 54, 120],
                            labels=["<18", "18-24", "25-34", "35-44", "45-54", "55+"]).astype("string")
    return u.sort_values("user_id").reset_index(drop=True)


def clean_items(items: pd.DataFrame) -> pd.DataFrame:
    """Parse dates, impute missing release dates, and derive genre fields."""
    it = items.copy()
    it["title"] = it["title"].astype("string").str.strip()
    it["release_date"] = pd.to_datetime(it["release_date"], format="%d-%b-%Y", errors="coerce")
    it["release_date_imputed"] = it["release_date"].isna().astype(int)
    it["release_date"] = it["release_date"].fillna(it["release_date"].median())
    it["release_year"] = it["release_date"].dt.year.astype("int64")
    names = [g for g in GENRES]
    flags = it[GENRE_COLS].to_numpy()
    it["genres"] = ["|".join(n for n, f in zip(names, row) if f == 1) or "unknown" for row in flags]
    it["n_genres"] = flags.sum(axis=1)
    it = it.drop(columns=["video_release_date"])
    return it.sort_values("item_id").reset_index(drop=True)


def clean_interactions(inter: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Add datetime, keep the latest rating per (user, item) pair; return (clean df, n dropped)."""
    x = inter.copy()
    x["datetime"] = pd.to_datetime(x["timestamp"], unit="s")
    x = x.sort_values(["timestamp", "user_id", "item_id"])
    before = len(x)
    x = x.drop_duplicates(subset=["user_id", "item_id"], keep="last")
    return x.reset_index(drop=True), before - len(x)


def time_based_split(inter: pd.DataFrame, test_fraction: float) -> tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp]:
    """Global temporal split: rows with timestamp <= cutoff -> train, > cutoff -> test."""
    cutoff_ts = int(inter["timestamp"].quantile(1 - test_fraction))
    train = inter[inter["timestamp"] <= cutoff_ts].reset_index(drop=True)
    test = inter[inter["timestamp"] > cutoff_ts].reset_index(drop=True)
    return train, test, pd.to_datetime(cutoff_ts, unit="s")


def run_preparation(cfg: Dict) -> Dict:
    """Run the preparation stage and return a summary dict."""
    proc, fmt = cfg["paths"]["processed"], cfg["storage"]["format"]
    users = clean_users(load_table(proc / "users_validated"))
    items = clean_items(load_table(proc / "items_validated"))
    inter, n_repeat = clean_interactions(load_table(proc / "interactions_validated"))
    train, test, cutoff = time_based_split(inter, cfg["preparation"]["test_fraction"])

    for name, df in [("users_clean", users), ("items_clean", items), ("interactions_clean", inter),
                     ("train", train), ("test", test)]:
        save_table(df, proc / name, fmt)

    train_users, train_items = set(train.user_id), set(train.item_id)
    meta = {
        "cutoff_datetime": str(cutoff),
        "train_rows": len(train), "test_rows": len(test),
        "train_period": [str(train.datetime.min()), str(train.datetime.max())],
        "test_period": [str(test.datetime.min()), str(test.datetime.max())],
        "users_in_test": int(test.user_id.nunique()),
        "cold_start_test_users": int(len(set(test.user_id) - train_users)),
        "cold_start_test_items": int(len(set(test.item_id) - train_items)),
        "repeat_ratings_removed": int(n_repeat),
        "imputed_occupation": int((users.occupation == "unknown").sum()),
        "imputed_release_dates": int(items.release_date_imputed.sum()),
    }
    save_json(meta, proc / "split_metadata.json")
    log.info("Preparation done. Time split at %s -> train %d rows, test %d rows (%d cold-start test users)",
             cutoff, len(train), len(test), meta["cold_start_test_users"])
    return meta
