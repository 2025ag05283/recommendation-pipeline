"""Feature registry - the single source of truth for feature metadata.

Every feature produced by ``build_features.py`` is declared here with its
entity, type, description and computation logic. The registry is used to
(1) generate the feature catalog (``reports/feature_catalog.md`` / ``.csv``)
and (2) check that the built feature tables contain exactly the declared columns.
"""
from __future__ import annotations

from typing import Dict, List

from src.ingestion.ingest import GENRES

# (name, entity, dtype, description, logic, used_by)
_FEATURES: List[tuple] = [
    # ---------------------------------------------------------------- user features
    ("user_age", "user", "int", "Age of the user", "users.age (validated)", "profile, hybrid"),
    ("user_gender", "user", "category", "Gender code (M/F)", "users.gender upper-cased", "profile"),
    ("user_occupation", "user", "category", "Occupation (missing -> 'unknown')", "users.occupation cleaned", "profile"),
    ("user_age_group", "user", "category", "Age bucket", "pd.cut(age, [0,17,24,34,44,54,120])", "profile"),
    ("user_activity_count", "user", "int", "Number of training interactions", "count(train rows per user)", "hybrid, analysis"),
    ("user_avg_rating", "user", "float", "Average rating given by the user", "mean(train.rating per user)", "hybrid, MF diagnostics"),
    ("user_rating_std", "user", "float", "Std. dev. of the user's ratings (0 if one rating)", "std(train.rating per user)", "analysis"),
    ("user_days_since_last_interaction", "user", "float", "Days between the user's last training interaction and the reference date (train cutoff)", "(cutoff - max(train.datetime)) in days", "recency, analysis"),
    ("user_active_span_days", "user", "float", "Days between first and last training interaction", "max(datetime) - min(datetime)", "analysis"),
    ("user_preferred_genre", "user", "category", "Genre the user rated most often among liked items (rating >= 4); falls back to all items", "argmax over genre counts of liked items", "explanations, hybrid"),
    ("user_n_distinct_genres", "user", "int", "Number of distinct genres the user interacted with", "count(genres with > 0 interactions)", "diversity analysis"),
    *[(f"user_genre_pref_{g}", "user", "float", f"Share of the user's liked items that belong to genre '{g}'",
       "sum(liked items with genre) / sum(all genre flags of liked items)", "hybrid (content affinity)") for g in GENRES],
    # ---------------------------------------------------------------- item features
    ("item_popularity", "item", "int", "Number of training ratings of the item", "count(train rows per item)", "popularity, hybrid"),
    ("item_popularity_rank", "item", "int", "Rank by popularity (1 = most popular)", "rank(item_popularity, desc)", "analysis"),
    ("item_user_share", "item", "float", "Fraction of all users who rated the item", "item_popularity / n_users", "novelty"),
    ("item_avg_rating", "item", "float", "Mean training rating of the item (NaN-safe: global mean if unrated)", "mean(train.rating per item)", "analysis"),
    ("item_bayesian_avg_rating", "item", "float", "Bayesian-average rating shrunk towards the global mean", "(m*mu + sum r)/(m + n), m = bayesian_prior_weight", "popularity rating prediction"),
    ("item_rating_std", "item", "float", "Std. dev. of the item's ratings", "std(train.rating per item)", "analysis"),
    ("item_release_year", "item", "int", "Release year (median-imputed when missing)", "year(release_date)", "content"),
    ("item_age_years", "item", "float", "Item age at the reference date", "(cutoff - release_date) / 365.25", "content"),
    ("item_days_since_last_interaction", "item", "float", "Days since the item was last rated (NaN-safe: large value if never)", "(cutoff - max(train.datetime per item)) in days", "trend analysis"),
    ("item_n_genres", "item", "int", "Number of genres of the item", "sum(genre flags)", "content"),
    *[(f"item_genre_{g}", "item", "int", f"1 if the item belongs to genre '{g}'", f"u.item genre flag '{g}'", "hybrid (content), diversity") for g in GENRES],
    # ---------------------------------------------------------- interaction features
    ("rating", "interaction", "int", "Explicit rating (label)", "train.rating", "MF training target"),
    ("is_relevant", "interaction", "int", "1 if rating >= relevance_threshold", "rating >= 4", "ranking labels"),
    ("rating_dev_from_user_avg", "interaction", "float", "Rating minus the user's average rating", "rating - user_avg_rating", "analysis, bias modelling"),
    ("rating_dev_from_item_avg", "interaction", "float", "Rating minus the item's average rating", "rating - item_avg_rating", "analysis, bias modelling"),
    ("interaction_recency_days", "interaction", "float", "Days between the interaction and the reference date", "(cutoff - datetime) in days", "recency weighting"),
    ("user_interaction_seq", "interaction", "int", "Order of the interaction in the user's history (1 = first)", "cumcount over user, sorted by time", "sequence analysis"),
]

KEYS = {"user": "user_id", "item": "item_id", "interaction": ["user_id", "item_id"]}


def feature_specs() -> List[Dict]:
    """Return the registry as a list of dicts (one per feature)."""
    cols = ["name", "entity", "dtype", "description", "logic", "used_by"]
    return [dict(zip(cols, f)) for f in _FEATURES]


def feature_names(entity: str) -> List[str]:
    """Names of all declared features for one entity (user | item | interaction)."""
    return [f[0] for f in _FEATURES if f[1] == entity]
