"""Stage 5 - Feature management (build + catalog).

Builds three feature tables from the TRAINING data, using the train cutoff as
the reference date ("as-of" date) so that features never see the test period:

* ``data/features/user_features``        (key: user_id)
* ``data/features/item_features``        (key: item_id)
* ``data/features/interaction_features`` (key: user_id, item_id)

and writes ``data/features/feature_metadata.json`` (version, as-of date, row
counts, column list) plus the feature catalog in ``reports/``. The same tables
are read by training *and* by prediction through :class:`src.features.store.FeatureStore`.
"""
from __future__ import annotations

import time
from typing import Dict

import numpy as np
import pandas as pd

from src.features.registry import KEYS, feature_names, feature_specs
from src.ingestion.ingest import GENRES
from src.utils.io import load_json, load_table, save_json, save_table
from src.utils.logger import get_logger

log = get_logger(__name__)
GENRE_COLS = [f"genre_{g}" for g in GENRES]
FEATURE_VERSION = "1.0.0"


def compute_genre_preferences(train: pd.DataFrame, items: pd.DataFrame, rel_threshold: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-user genre preference shares (reusable: also called by the hybrid model's tuning step).

    Shares are computed from *liked* items (rating >= threshold); users who liked nothing
    fall back to all their rated items. Returns (shares indexed by user_id with columns
    ``user_genre_pref_<genre>``, raw genre counts over all rated items).
    """
    x = train[["user_id", "item_id", "rating"]].merge(items[["item_id"] + GENRE_COLS], on="item_id")
    liked_counts = x[x["rating"] >= rel_threshold].groupby("user_id")[GENRE_COLS].sum()
    all_counts = x.groupby("user_id")[GENRE_COLS].sum()
    counts = all_counts.copy()
    counts.loc[liked_counts.index] = liked_counts
    zero = counts.sum(axis=1) == 0
    counts.loc[zero] = all_counts.loc[zero]
    shares = counts.div(counts.sum(axis=1).replace(0, 1), axis=0)
    shares.columns = [f"user_genre_pref_{g}" for g in GENRES]
    return shares, all_counts


def build_user_features(users: pd.DataFrame, train: pd.DataFrame, items: pd.DataFrame,
                        asof: pd.Timestamp, rel_threshold: int) -> pd.DataFrame:
    """Profile + behavioural + genre-preference features for every known user."""
    f = users[["user_id"]].copy()
    f["user_age"] = users["age"].astype(int)
    f["user_gender"] = users["gender"]
    f["user_occupation"] = users["occupation"]
    f["user_age_group"] = users["age_group"]

    g = train.groupby("user_id")
    act = g.agg(user_activity_count=("item_id", "size"), user_avg_rating=("rating", "mean"),
                user_rating_std=("rating", "std"), first=("datetime", "min"), last=("datetime", "max"))
    act["user_rating_std"] = act["user_rating_std"].fillna(0.0)
    act["user_days_since_last_interaction"] = (asof - act["last"]).dt.total_seconds() / 86400
    act["user_active_span_days"] = (act["last"] - act["first"]).dt.total_seconds() / 86400
    f = f.merge(act.drop(columns=["first", "last"]).reset_index(), on="user_id", how="left")

    shares, all_counts = compute_genre_preferences(train, items, rel_threshold)
    pref = shares.idxmax(axis=1).str.replace("user_genre_pref_", "", regex=False)
    extra = pd.DataFrame({"user_preferred_genre": pref, "user_n_distinct_genres": (all_counts > 0).sum(axis=1)})
    f = f.merge(extra.join(shares).reset_index(), on="user_id", how="left")

    # users with no training history (cold start): neutral defaults, explicitly typed
    f["user_activity_count"] = f["user_activity_count"].fillna(0).astype(int)
    f["user_avg_rating"] = f["user_avg_rating"].fillna(train["rating"].mean())
    for c in ["user_rating_std", "user_active_span_days", "user_n_distinct_genres"] + list(shares.columns):
        f[c] = f[c].fillna(0)
    f["user_n_distinct_genres"] = f["user_n_distinct_genres"].astype(int)
    f["user_days_since_last_interaction"] = f["user_days_since_last_interaction"].fillna(-1.0)  # -1 = never active
    f["user_preferred_genre"] = f["user_preferred_genre"].fillna("unknown")
    return f


def build_item_features(items: pd.DataFrame, train: pd.DataFrame, n_users: int, asof: pd.Timestamp,
                        prior_weight: float) -> pd.DataFrame:
    """Popularity, rating and content features for every known item."""
    mu = train["rating"].mean()
    g = train.groupby("item_id")
    st = g.agg(item_popularity=("rating", "size"), s=("rating", "sum"), item_avg_rating=("rating", "mean"),
               item_rating_std=("rating", "std"), last=("datetime", "max")).reset_index()
    f = items[["item_id"]].merge(st, on="item_id", how="left")
    f["item_popularity"] = f["item_popularity"].fillna(0).astype(int)
    f["s"] = f["s"].fillna(0)
    f["item_popularity_rank"] = f["item_popularity"].rank(ascending=False, method="min").astype(int)
    f["item_user_share"] = f["item_popularity"] / n_users
    f["item_avg_rating"] = f["item_avg_rating"].fillna(mu)
    f["item_bayesian_avg_rating"] = (prior_weight * mu + f["s"]) / (prior_weight + f["item_popularity"])
    f["item_rating_std"] = f["item_rating_std"].fillna(0.0)
    f["item_release_year"] = items["release_year"].astype(int).to_numpy()
    f["item_age_years"] = ((asof - pd.to_datetime(items["release_date"])).dt.days / 365.25).to_numpy()
    f["item_days_since_last_interaction"] = ((asof - f["last"]).dt.total_seconds() / 86400).fillna(9999.0)
    f["item_n_genres"] = items["n_genres"].astype(int).to_numpy()
    for gname in GENRES:
        f[f"item_genre_{gname}"] = items[f"genre_{gname}"].astype(int).to_numpy()
    return f.drop(columns=["s", "last"])


def build_interaction_features(train: pd.DataFrame, user_f: pd.DataFrame, item_f: pd.DataFrame,
                               asof: pd.Timestamp, rel_threshold: int) -> pd.DataFrame:
    """Row-level features for each training interaction."""
    x = train[["user_id", "item_id", "rating", "datetime"]].copy()
    x = x.merge(user_f[["user_id", "user_avg_rating"]], on="user_id").merge(item_f[["item_id", "item_avg_rating"]], on="item_id")
    x["is_relevant"] = (x["rating"] >= rel_threshold).astype(int)
    x["rating_dev_from_user_avg"] = x["rating"] - x["user_avg_rating"]
    x["rating_dev_from_item_avg"] = x["rating"] - x["item_avg_rating"]
    x["interaction_recency_days"] = (asof - x["datetime"]).dt.total_seconds() / 86400
    x = x.sort_values(["user_id", "datetime"])
    x["user_interaction_seq"] = x.groupby("user_id").cumcount() + 1
    return x.drop(columns=["user_avg_rating", "item_avg_rating", "datetime"]).reset_index(drop=True)


def write_catalog(cfg: Dict, stats: Dict[str, pd.DataFrame]) -> None:
    """Write the feature catalog (Markdown + CSV) with metadata and simple statistics."""
    specs = pd.DataFrame(feature_specs())
    rows = []
    for s in specs.to_dict("records"):
        df = stats[s["entity"]]
        col = df[s["name"]]
        if pd.api.types.is_numeric_dtype(col):
            example = f"min={col.min():.3g}, mean={col.mean():.3g}, max={col.max():.3g}"
        else:
            example = f"{col.nunique()} values, top='{col.mode().iloc[0]}'"
        rows.append({**s, "key": str(KEYS[s["entity"]]), "null_pct": round(100 * col.isna().mean(), 2), "stats": example})
    cat = pd.DataFrame(rows)
    cat.to_csv(cfg["paths"]["reports"] / "feature_catalog.csv", index=False)

    lines = ["# Feature Catalog", "",
             f"Feature version **{FEATURE_VERSION}** - all features are computed **as of the train cutoff** "
             "from training data only (no test leakage). Stored in `data/features/` and served to training "
             "and prediction through `src/features/store.py::FeatureStore`.", ""]
    for entity in ["user", "item", "interaction"]:
        sub = cat[cat.entity == entity]
        lines += [f"## {entity.capitalize()} features (key: `{KEYS[entity]}`) - {len(sub)} features", "",
                  "| Feature | Type | Description | Computation | Used by | Null % | Statistics |",
                  "|---|---|---|---|---|---:|---|"]
        for r in sub.to_dict("records"):
            lines.append(f"| `{r['name']}` | {r['dtype']} | {r['description']} | {r['logic'].replace('|', '/')} | "
                         f"{r['used_by']} | {r['null_pct']} | {r['stats']} |")
        lines.append("")
    (cfg["paths"]["reports"] / "feature_catalog.md").write_text("\n".join(lines), encoding="utf-8")


def run_feature_engineering(cfg: Dict) -> Dict:
    """Build, validate against the registry, store and catalog all features."""
    proc, feat, fmt = cfg["paths"]["processed"], cfg["paths"]["features"], cfg["storage"]["format"]
    rel = cfg["evaluation"]["relevance_threshold"]
    train = load_table(proc / "train")
    train["datetime"] = pd.to_datetime(train["datetime"])
    users, items = load_table(proc / "users_clean"), load_table(proc / "items_clean")
    items["release_date"] = pd.to_datetime(items["release_date"])
    asof = pd.Timestamp(load_json(proc / "split_metadata.json")["cutoff_datetime"])

    uf = build_user_features(users, train, items, asof, rel)
    itf = build_item_features(items, train, len(users), asof, cfg["features"]["bayesian_prior_weight"])
    xf = build_interaction_features(train, uf, itf, asof, rel)

    # contract check: built columns == registry declaration
    for name, df, key in [("user", uf, ["user_id"]), ("item", itf, ["item_id"]), ("interaction", xf, ["user_id", "item_id"])]:
        declared, built = set(feature_names(name)), set(df.columns) - set(key)
        if declared != built:
            raise ValueError(f"{name} features differ from registry: missing={declared - built}, extra={built - declared}")

    for name, df in [("user_features", uf), ("item_features", itf), ("interaction_features", xf)]:
        save_table(df, feat / name, fmt)
    write_catalog(cfg, {"user": uf, "item": itf, "interaction": xf})
    meta = {"feature_version": FEATURE_VERSION, "as_of": str(asof), "built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "tables": {"user_features": {"rows": len(uf), "columns": list(uf.columns)},
                       "item_features": {"rows": len(itf), "columns": list(itf.columns)},
                       "interaction_features": {"rows": len(xf), "columns": list(xf.columns)}},
            "n_features": len(feature_specs())}
    save_json(meta, feat / "feature_metadata.json")
    log.info("Features built: %d users x %d cols, %d items x %d cols, %d interactions (%d registered features)",
             len(uf), uf.shape[1], len(itf), itf.shape[1], len(xf), len(feature_specs()))
    return {k: v["rows"] for k, v in meta["tables"].items()} | {"n_features": meta["n_features"]}
