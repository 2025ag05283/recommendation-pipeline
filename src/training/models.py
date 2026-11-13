"""Recommender models.

All models share one interface (:class:`BaseRecommender`):

* ``fit(train, store)``            - learn from training interactions + feature store;
* ``score(user_id) -> np.ndarray`` - a score for every item (aligned with ``store.item_ids``);
* ``recommend(user_id, k)``        - top-k unseen items;
* ``predict_rating(users, items)`` - explicit rating predictions (if supported).

Models:
1. :class:`PopularityRecommender`        - baseline: most-rated items; ratings = Bayesian item average.
2. :class:`MatrixFactorizationRecommender` - biased matrix factorisation (Funk-SVD) trained with SGD.
3. :class:`HybridRecommender`            - MF score + content (genre affinity from the feature store)
                                            + popularity prior, blend weights tuned on a time-based
                                            validation split carved from TRAIN.
Unknown (cold-start) users fall back to popularity in every model.
"""
from __future__ import annotations

import pickle
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from src.features.store import FeatureStore
from src.utils.logger import get_logger

log = get_logger(__name__)


class BaseRecommender:
    """Common functionality: item indexing, seen-item exclusion, top-k, persistence."""

    name = "base"

    def __init__(self) -> None:
        self.item_ids: np.ndarray = np.array([])
        self.item_pos: Dict[int, int] = {}
        self.seen: Dict[int, np.ndarray] = {}
        self.popularity: np.ndarray = np.array([])
        self.fit_seconds: float = 0.0

    # ---------------------------------------------------------------- setup
    def _setup(self, train: pd.DataFrame, store: FeatureStore) -> None:
        """Index items exactly like the feature store and remember what each user has seen."""
        self.item_ids = store.item_ids
        self.item_pos = {int(i): p for p, i in enumerate(self.item_ids)}
        pos = train["item_id"].map(self.item_pos).to_numpy()
        self.seen = {int(u): pos[idx] for u, idx in train.groupby("user_id").indices.items()}
        self.popularity = store.item_popularity()

    def is_known_user(self, user_id: int) -> bool:
        """True if the user has training history."""
        return int(user_id) in self.seen

    # -------------------------------------------------------------- scoring
    def score(self, user_id: int) -> np.ndarray:  # pragma: no cover - overridden
        """Score every item for the user (higher = better)."""
        raise NotImplementedError

    def recommend(self, user_id: int, k: int = 10, exclude_seen: bool = True) -> List[int]:
        """Return the ids of the top-k items for a user, excluding items already rated in train."""
        s = self.score(user_id).astype(float).copy()
        if exclude_seen and int(user_id) in self.seen:
            s[self.seen[int(user_id)]] = -np.inf
        top = np.argpartition(-s, k)[:k]
        top = top[np.argsort(-s[top])]
        return [int(self.item_ids[i]) for i in top]

    def predict_rating(self, user_ids: np.ndarray, item_ids: np.ndarray) -> Optional[np.ndarray]:
        """Explicit rating predictions; ``None`` if the model is ranking-only."""
        return None

    # ---------------------------------------------------------- persistence
    def save(self, path: Path) -> Path:
        """Pickle the fitted model."""
        with open(path, "wb") as fh:
            pickle.dump(self, fh)
        return path

    @staticmethod
    def load(path: Path) -> "BaseRecommender":
        """Load a pickled model."""
        with open(path, "rb") as fh:
            return pickle.load(fh)

    def params(self) -> Dict:
        """Hyper-parameters to log."""
        return {}


# =========================================================================== 1
class PopularityRecommender(BaseRecommender):
    """Non-personalised baseline: rank items by number of training ratings."""

    name = "popularity"

    def fit(self, train: pd.DataFrame, store: FeatureStore) -> "PopularityRecommender":
        """Store popularity counts and Bayesian-average ratings from the feature store."""
        t0 = time.time()
        self._setup(train, store)
        self.bayes_rating = store.item["item_bayesian_avg_rating"].to_numpy(dtype=float)
        self.global_mean = float(train["rating"].mean())
        self.fit_seconds = time.time() - t0
        return self

    def score(self, user_id: int) -> np.ndarray:
        """Same popularity score for every user."""
        return self.popularity

    def predict_rating(self, user_ids, item_ids) -> np.ndarray:
        """Predict the item's Bayesian-average rating (global mean for unknown items)."""
        pos = pd.Series(item_ids).map(self.item_pos)
        out = np.full(len(pos), self.global_mean)
        ok = pos.notna().to_numpy()
        out[ok] = self.bayes_rating[pos[ok].astype(int).to_numpy()]
        return out

    def params(self) -> Dict:
        return {"model": self.name, "ranking": "training interaction count", "rating_prediction": "bayesian item average"}


# =========================================================================== 2
class MatrixFactorizationRecommender(BaseRecommender):
    """Biased matrix factorisation  r_ui ~ mu + b_u + b_i + p_u . q_i , trained with SGD."""

    name = "matrix_factorization"

    def __init__(self, n_factors: int = 40, n_epochs: int = 30, learning_rate: float = 0.01,
                 regularization: float = 0.05, min_item_support: int = 5, seed: int = 42) -> None:
        super().__init__()
        self.n_factors, self.n_epochs = n_factors, n_epochs
        self.lr, self.reg, self.min_item_support, self.seed = learning_rate, regularization, min_item_support, seed
        self.train_rmse_history: List[float] = []

    def fit(self, train: pd.DataFrame, store: FeatureStore) -> "MatrixFactorizationRecommender":
        """Learn global mean, user/item biases and latent factors with per-sample SGD."""
        t0 = time.time()
        self._setup(train, store)
        rng = np.random.default_rng(self.seed)
        self.user_pos = {int(u): p for p, u in enumerate(sorted(train["user_id"].unique()))}
        n_u, n_i = len(self.user_pos), len(self.item_ids)
        u = train["user_id"].map(self.user_pos).to_numpy()
        i = train["item_id"].map(self.item_pos).to_numpy()
        r = train["rating"].to_numpy(dtype=float)
        self.mu = float(r.mean())
        self.bu, self.bi = np.zeros(n_u), np.zeros(n_i)
        self.P = rng.normal(0, 0.1, (n_u, self.n_factors))
        self.Q = rng.normal(0, 0.1, (n_i, self.n_factors))
        lr, reg = self.lr, self.reg
        for epoch in range(self.n_epochs):
            for idx in rng.permutation(len(r)):                  # one SGD step per rating
                uu, ii = u[idx], i[idx]
                pu, qi = self.P[uu], self.Q[ii]
                err = r[idx] - (self.mu + self.bu[uu] + self.bi[ii] + pu @ qi)
                self.bu[uu] += lr * (err - reg * self.bu[uu])
                self.bi[ii] += lr * (err - reg * self.bi[ii])
                self.P[uu] = pu + lr * (err * qi - reg * pu)
                self.Q[ii] = qi + lr * (err * pu - reg * qi)
            pred = self.mu + self.bu[u] + self.bi[i] + np.einsum("ij,ij->i", self.P[u], self.Q[i])
            self.train_rmse_history.append(float(np.sqrt(np.mean((r - pred) ** 2))))
            if (epoch + 1) % 10 == 0 or epoch == 0:
                log.info("  MF epoch %2d/%d  train RMSE = %.4f", epoch + 1, self.n_epochs, self.train_rmse_history[-1])
        self.support_mask = self.popularity >= self.min_item_support
        self.fit_seconds = time.time() - t0
        return self

    def predict_all(self, user_id: int) -> np.ndarray:
        """Predicted rating for every item (bias-only prediction for unknown users)."""
        p = self.user_pos.get(int(user_id))
        if p is None:
            return self.mu + self.bi
        return self.mu + self.bu[p] + self.bi + self.Q @ self.P[p]

    def score(self, user_id: int) -> np.ndarray:
        """Predicted ratings; items with too little support are pushed to the bottom; cold users -> popularity."""
        if not self.is_known_user(user_id):
            return self.popularity
        s = self.predict_all(user_id)
        return np.where(self.support_mask, s, -1e9)

    def predict_rating(self, user_ids, item_ids) -> np.ndarray:
        """Clipped rating predictions for (user, item) pairs; unknown users/items use the available biases."""
        out = np.empty(len(user_ids))
        for n, (uid, iid) in enumerate(zip(user_ids, item_ids)):
            pu, qi = self.user_pos.get(int(uid)), self.item_pos.get(int(iid))
            val = self.mu
            if pu is not None:
                val += self.bu[pu]
            if qi is not None:
                val += self.bi[qi]
            if pu is not None and qi is not None:
                val += self.P[pu] @ self.Q[qi]
            out[n] = val
        return np.clip(out, 1, 5)

    def params(self) -> Dict:
        return {"model": self.name, "n_factors": self.n_factors, "n_epochs": self.n_epochs,
                "learning_rate": self.lr, "regularization": self.reg, "min_item_support": self.min_item_support,
                "seed": self.seed}


# =========================================================================== 3
def _zscore(x: np.ndarray) -> np.ndarray:
    """Standardise a score vector (0 vector if constant)."""
    sd = x.std()
    return (x - x.mean()) / sd if sd > 0 else np.zeros_like(x)


class HybridRecommender(BaseRecommender):
    """score = z(MF) + w_content * z(genre affinity) + w_pop * z(log popularity)."""

    name = "hybrid"

    def __init__(self, mf: MatrixFactorizationRecommender, w_content: float = 0.5, w_pop: float = 0.5) -> None:
        super().__init__()
        self.mf, self.w_content, self.w_pop = mf, w_content, w_pop

    def fit(self, train: pd.DataFrame, store: FeatureStore, user_prefs: Optional[pd.DataFrame] = None) -> "HybridRecommender":
        """Attach the (already trained) MF model and the content features.

        ``user_prefs`` (users x genres) defaults to the feature store; the tuning step passes
        preferences computed with the *same feature function* on the inner-train data.
        """
        t0 = time.time()
        self._setup(train, store)
        G = store.item_genre_matrix()
        self.item_genre_unit = G / np.maximum(np.linalg.norm(G, axis=1, keepdims=True), 1e-9)
        if user_prefs is None:
            ids = store.user.index.to_numpy()
            prefs = store.user_genre_preferences(ids)
        else:
            ids, prefs = user_prefs.index.to_numpy(), user_prefs.to_numpy(dtype=float)
        norms = np.maximum(np.linalg.norm(prefs, axis=1, keepdims=True), 1e-9)
        self.user_pref_unit = {int(u): v for u, v in zip(ids, prefs / norms)}
        self.log_pop_z = _zscore(np.log1p(self.popularity))
        self.fit_seconds = time.time() - t0 + self.mf.fit_seconds
        return self

    def score(self, user_id: int) -> np.ndarray:
        """Blend MF, content affinity and popularity; cold users -> popularity."""
        if not self.mf.is_known_user(user_id):
            return self.popularity
        mf = self.mf.predict_all(user_id)
        content = self.item_genre_unit @ self.user_pref_unit.get(int(user_id), np.zeros(self.item_genre_unit.shape[1]))
        s = _zscore(mf) + self.w_content * _zscore(content) + self.w_pop * self.log_pop_z
        return np.where(self.mf.support_mask, s, -1e9)

    def params(self) -> Dict:
        return {"model": self.name, "w_content": self.w_content, "w_pop": self.w_pop,
                "base_mf_factors": self.mf.n_factors}
