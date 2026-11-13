"""Stage 4 - Data transformation (derived datasets).

All derived datasets are computed from the **training split only**, so no
information from the future (test period) leaks into features or models.

Outputs in ``data/processed/``:

* ``user_activity_summary`` - per-user counts, rating stats and activity span;
* ``item_popularity``       - per-item counts, mean / Bayesian-average rating, rank;
* ``item_genre_matrix``     - one-hot genre matrix (items x 19 genres);
* ``user_item_matrix.npz``  - sparse users x items rating matrix (+ index maps JSON).
"""
from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd
from scipy import sparse

from src.ingestion.ingest import GENRES
from src.utils.io import load_table, save_json, save_table
from src.utils.logger import get_logger

log = get_logger(__name__)
GENRE_COLS = [f"genre_{g}" for g in GENRES]


def user_activity_summary(train: pd.DataFrame) -> pd.DataFrame:
    """Per-user activity statistics from the training interactions."""
    g = train.groupby("user_id")
    out = g.agg(n_interactions=("item_id", "size"), avg_rating=("rating", "mean"),
                rating_std=("rating", "std"), first_interaction=("datetime", "min"),
                last_interaction=("datetime", "max")).reset_index()
    out["rating_std"] = out["rating_std"].fillna(0.0)
    out["active_span_days"] = (out["last_interaction"] - out["first_interaction"]).dt.total_seconds() / 86400
    return out


def item_popularity(train: pd.DataFrame, n_users: int, prior_weight: float) -> pd.DataFrame:
    """Per-item popularity: counts, share of users, mean and Bayesian-average rating, rank."""
    mu = train["rating"].mean()
    g = train.groupby("item_id")
    out = g.agg(n_ratings=("rating", "size"), sum_rating=("rating", "sum"), avg_rating=("rating", "mean"),
                rating_std=("rating", "std"), last_interaction=("datetime", "max")).reset_index()
    out["rating_std"] = out["rating_std"].fillna(0.0)
    out["user_share"] = out["n_ratings"] / n_users
    # Bayesian average shrinks items with few ratings towards the global mean
    out["bayesian_avg_rating"] = (prior_weight * mu + out["sum_rating"]) / (prior_weight + out["n_ratings"])
    out["popularity_rank"] = out["n_ratings"].rank(ascending=False, method="min").astype(int)
    return out.drop(columns=["sum_rating"]).sort_values("popularity_rank").reset_index(drop=True)


def build_user_item_matrix(train: pd.DataFrame, user_ids, item_ids) -> tuple[sparse.csr_matrix, dict, dict]:
    """Sparse rating matrix over ALL known users/items (rows=users, cols=items)."""
    u_index = {int(u): i for i, u in enumerate(sorted(user_ids))}
    i_index = {int(it): j for j, it in enumerate(sorted(item_ids))}
    rows = train["user_id"].map(u_index).to_numpy()
    cols = train["item_id"].map(i_index).to_numpy()
    mat = sparse.csr_matrix((train["rating"].to_numpy(dtype=np.float32), (rows, cols)),
                            shape=(len(u_index), len(i_index)))
    return mat, u_index, i_index


def run_transformation(cfg: Dict) -> Dict:
    """Run the transformation stage and return a summary dict."""
    proc, fmt = cfg["paths"]["processed"], cfg["storage"]["format"]
    train = load_table(proc / "train", parse_dates=["datetime"])
    train["datetime"] = pd.to_datetime(train["datetime"])
    users = load_table(proc / "users_clean")
    items = load_table(proc / "items_clean")

    uas = user_activity_summary(train)
    pop = item_popularity(train, len(users), cfg["features"]["bayesian_prior_weight"])
    genre = items[["item_id"] + GENRE_COLS].copy()
    mat, u_index, i_index = build_user_item_matrix(train, users.user_id, items.item_id)

    save_table(uas, proc / "user_activity_summary", fmt)
    save_table(pop, proc / "item_popularity", fmt)
    save_table(genre, proc / "item_genre_matrix", fmt)
    sparse.save_npz(proc / "user_item_matrix.npz", mat)
    save_json({"user_index": u_index, "item_index": i_index}, proc / "user_item_matrix_index.json")

    density = mat.nnz / (mat.shape[0] * mat.shape[1])
    summary = {"users_with_activity": len(uas), "items_with_ratings": len(pop),
               "matrix_shape": list(mat.shape), "matrix_nnz": int(mat.nnz), "matrix_density": round(density, 5)}
    save_json(summary, proc / "transformation_summary.json")
    log.info("Transformation done: user-item matrix %s, density %.4f", mat.shape, density)
    return summary
