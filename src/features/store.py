"""A lightweight, file-based feature store.

The same class is used at training time (``src/training``) and at prediction
time (``recommend.py``), which guarantees that both use identical feature
definitions and values (no training/serving skew).
"""
from __future__ import annotations

from typing import Dict, Iterable, List

import numpy as np
import pandas as pd

from src.ingestion.ingest import GENRES
from src.utils.io import load_json, load_table


class FeatureStore:
    """Read-only access to the user / item / interaction feature tables."""

    def __init__(self, cfg: Dict):
        feat = cfg["paths"]["features"]
        self.metadata = load_json(feat / "feature_metadata.json")
        self.user = load_table(feat / "user_features").set_index("user_id").sort_index()
        self.item = load_table(feat / "item_features").set_index("item_id").sort_index()
        self.interaction = load_table(feat / "interaction_features")

    # -------------------------------------------------------------- lookups
    def get_user_features(self, user_ids: Iterable[int], columns: List[str] | None = None) -> pd.DataFrame:
        """Feature rows for the given users (unknown users -> NaN rows)."""
        df = self.user.reindex(list(user_ids))
        return df[columns] if columns else df

    def get_item_features(self, item_ids: Iterable[int], columns: List[str] | None = None) -> pd.DataFrame:
        """Feature rows for the given items (unknown items -> NaN rows)."""
        df = self.item.reindex(list(item_ids))
        return df[columns] if columns else df

    # --------------------------------------------------------- matrix views
    @property
    def item_ids(self) -> np.ndarray:
        """All item ids in a fixed (sorted) order - the column order used by every model."""
        return self.item.index.to_numpy()

    def item_genre_matrix(self) -> np.ndarray:
        """Items x genres 0/1 matrix aligned with :attr:`item_ids`."""
        return self.item[[f"item_genre_{g}" for g in GENRES]].to_numpy(dtype=float)

    def user_genre_preferences(self, user_ids: Iterable[int]) -> np.ndarray:
        """Users x genres preference-share matrix (zeros for unknown users)."""
        cols = [f"user_genre_pref_{g}" for g in GENRES]
        return self.user.reindex(list(user_ids))[cols].fillna(0.0).to_numpy(dtype=float)

    def item_popularity(self) -> np.ndarray:
        """Training popularity count aligned with :attr:`item_ids`."""
        return self.item["item_popularity"].to_numpy(dtype=float)
