"""Prediction / serving entry point.

Loads a trained model and the SAME feature store used in training, and prints
top-K recommendations for a user, with feature-based context.

    python recommend.py --user 42
    python recommend.py --user 42 --model matrix_factorization --k 5
"""
from __future__ import annotations

import argparse

from src.features.store import FeatureStore
from src.training.models import BaseRecommender
from src.utils.config import load_config
from src.utils.io import load_table


def main() -> None:
    """Parse arguments, load model + features, print recommendations."""
    p = argparse.ArgumentParser(description="Recommend items for a user")
    p.add_argument("--user", type=int, required=True, help="user_id")
    p.add_argument("--model", default="hybrid", choices=["popularity", "matrix_factorization", "hybrid"])
    p.add_argument("--k", type=int, default=10)
    args = p.parse_args()

    cfg = load_config()
    store = FeatureStore(cfg)                                    # identical features to training
    model = BaseRecommender.load(cfg["paths"]["models"] / f"{args.model}.pkl")
    items = load_table(cfg["paths"]["processed"] / "items_clean").set_index("item_id")

    uf = store.get_user_features([args.user])
    known = model.is_known_user(args.user)
    print(f"\nUser {args.user} | known user: {known}")
    if uf.notna().any(axis=None):
        r = uf.iloc[0]
        print(f"  activity={int(r.user_activity_count)} ratings, avg rating={r.user_avg_rating:.2f}, "
              f"preferred genre={r.user_preferred_genre}, days since last interaction={r.user_days_since_last_interaction:.0f}")
    if not known:
        print("  cold-start user -> popularity fallback")
    recs = model.recommend(args.user, args.k)
    itf = store.get_item_features(recs, ["item_popularity", "item_bayesian_avg_rating"])
    print(f"\nTop-{args.k} recommendations from '{args.model}':")
    for rank, iid in enumerate(recs, 1):
        print(f"  {rank:2d}. [{iid:4d}] {items.loc[iid, 'title'][:55]:55s} genres={items.loc[iid, 'genres']:<25s} "
              f"pop={int(itf.loc[iid, 'item_popularity']):4d} avg={itf.loc[iid, 'item_bayesian_avg_rating']:.2f}")


if __name__ == "__main__":
    main()
