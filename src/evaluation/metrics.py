"""Recommendation metrics (pure functions, unit-tested in ``tests/test_metrics.py``).

Ranking metrics use binary relevance: a test item is *relevant* to a user when
its test rating >= relevance threshold (default 4).
"""
from __future__ import annotations

from itertools import combinations
from typing import Dict, Iterable, List, Sequence, Set

import numpy as np


def precision_at_k(recs: Sequence[int], relevant: Set[int], k: int) -> float:
    """Fraction of the top-k recommendations that are relevant."""
    return len(set(recs[:k]) & relevant) / k


def recall_at_k(recs: Sequence[int], relevant: Set[int], k: int) -> float:
    """Fraction of the user's relevant items that appear in the top-k."""
    return len(set(recs[:k]) & relevant) / len(relevant) if relevant else 0.0


def ndcg_at_k(recs: Sequence[int], relevant: Set[int], k: int) -> float:
    """Normalised discounted cumulative gain with binary gains (1 if relevant)."""
    dcg = sum(1.0 / np.log2(pos + 2) for pos, item in enumerate(recs[:k]) if item in relevant)
    idcg = sum(1.0 / np.log2(pos + 2) for pos in range(min(len(relevant), k)))
    return dcg / idcg if idcg > 0 else 0.0


def hit_rate_at_k(recs: Sequence[int], relevant: Set[int], k: int) -> float:
    """1 if at least one relevant item is in the top-k, else 0."""
    return float(bool(set(recs[:k]) & relevant))


def catalog_coverage(all_recs: Iterable[Sequence[int]], n_items: int) -> float:
    """Share of the catalogue that appears in at least one user's top-k list."""
    unique = set()
    for r in all_recs:
        unique.update(r)
    return len(unique) / n_items if n_items else 0.0


def intra_list_diversity(recs: Sequence[int], item_genres: Dict[int, frozenset]) -> float:
    """Mean pairwise genre dissimilarity (1 - Jaccard) inside one recommendation list."""
    pairs = list(combinations(recs, 2))
    if not pairs:
        return 0.0
    dist = []
    for a, b in pairs:
        ga, gb = item_genres.get(a, frozenset()), item_genres.get(b, frozenset())
        union = len(ga | gb)
        dist.append(1.0 - (len(ga & gb) / union if union else 0.0))
    return float(np.mean(dist))


def novelty(recs: Sequence[int], item_user_share: Dict[int, float], eps: float = 1e-6) -> float:
    """Mean self-information -log2(p(item)) of the recommended items (higher = less mainstream)."""
    return float(np.mean([-np.log2(max(item_user_share.get(i, 0.0), eps)) for i in recs])) if len(recs) else 0.0


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root mean squared error."""
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean absolute error."""
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def ranking_report(recs_by_user: Dict[int, List[int]], relevant_by_user: Dict[int, Set[int]], k: int) -> Dict[str, float]:
    """Average precision/recall/NDCG/hit-rate@k over users that have at least one relevant item."""
    users = [u for u in recs_by_user if relevant_by_user.get(u)]
    if not users:
        return {f"precision_at_{k}": 0.0, f"recall_at_{k}": 0.0, f"ndcg_at_{k}": 0.0, f"hit_rate_at_{k}": 0.0, "n_users": 0}
    return {
        f"precision_at_{k}": float(np.mean([precision_at_k(recs_by_user[u], relevant_by_user[u], k) for u in users])),
        f"recall_at_{k}": float(np.mean([recall_at_k(recs_by_user[u], relevant_by_user[u], k) for u in users])),
        f"ndcg_at_{k}": float(np.mean([ndcg_at_k(recs_by_user[u], relevant_by_user[u], k) for u in users])),
        f"hit_rate_at_{k}": float(np.mean([hit_rate_at_k(recs_by_user[u], relevant_by_user[u], k) for u in users])),
        "n_users": len(users),
    }
