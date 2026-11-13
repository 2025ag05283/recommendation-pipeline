"""Unit tests for the evaluation metrics (hand-computed expected values)."""
import math

from src.evaluation.metrics import (catalog_coverage, intra_list_diversity, mae, ndcg_at_k, precision_at_k,
                                    recall_at_k, rmse)


def test_precision_and_recall():
    recs, relevant = [1, 2, 3, 4, 5], {2, 5, 9}
    assert precision_at_k(recs, relevant, 5) == 2 / 5
    assert recall_at_k(recs, relevant, 5) == 2 / 3


def test_ndcg_perfect_and_partial():
    assert ndcg_at_k([1, 2, 3], {1, 2, 3}, 3) == 1.0
    # one relevant item at position 2 -> DCG = 1/log2(3); IDCG = 1
    assert math.isclose(ndcg_at_k([9, 1, 8], {1}, 3), 1 / math.log2(3))
    assert ndcg_at_k([7, 8], {1}, 2) == 0.0


def test_coverage_and_diversity():
    assert catalog_coverage([[1, 2], [2, 3]], 10) == 0.3
    genres = {1: frozenset({"a"}), 2: frozenset({"a"}), 3: frozenset({"b"})}
    assert intra_list_diversity([1, 2], genres) == 0.0          # identical genres
    assert intra_list_diversity([1, 3], genres) == 1.0          # disjoint genres


def test_rating_errors():
    assert rmse([1, 2, 3], [1, 2, 5]) == math.sqrt(4 / 3)
    assert mae([1, 2, 3], [2, 2, 2]) == 2 / 3
