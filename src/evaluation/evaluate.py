"""Stage 7 - Model evaluation.

Evaluates every trained model on the TEST period (interactions after the train
cutoff):

* **Ranking** (warm users = users with training history and >= 1 relevant test
  item): Precision@K, Recall@K, NDCG@K, HitRate@K.
* **Beyond-accuracy**: catalogue Coverage, intra-list Diversity (genre
  dissimilarity) and Novelty.
* **Rating prediction** (models that predict ratings): RMSE and MAE over all
  test interactions.
* **Cold-start users** are reported separately (every model falls back to popularity).

Outputs: ``reports/model_comparison.{csv,md}``, ``reports/figures/model_comparison.png``,
``reports/per_user_metrics.csv``, ``reports/sample_recommendations.csv``;
test metrics are appended to each model's MLflow run.
"""
from __future__ import annotations

from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.evaluation.metrics import (catalog_coverage, intra_list_diversity, mae, ndcg_at_k, novelty,
                                    precision_at_k, ranking_report, recall_at_k, rmse)
from src.features.store import FeatureStore
from src.ingestion.ingest import GENRES
from src.training.models import BaseRecommender
from src.utils.io import load_json, load_table, save_json
from src.utils.logger import get_logger
from src.utils.tracking import ExperimentTracker

log = get_logger(__name__)
MODEL_NAMES = ["popularity", "matrix_factorization", "hybrid"]
PRETTY = {"popularity": "Popularity (baseline)", "matrix_factorization": "Matrix Factorization (SVD)",
          "hybrid": "Hybrid (MF + content + popularity)"}


def _plot(results: pd.DataFrame, k: int, path) -> None:
    """Grouped bar chart of the main metrics for all models."""
    metrics = [f"precision_at_{k}", f"recall_at_{k}", f"ndcg_at_{k}", "coverage", "diversity"]
    labels = [f"Precision@{k}", f"Recall@{k}", f"NDCG@{k}", "Coverage", "Diversity"]
    x = np.arange(len(metrics))
    w = 0.8 / len(results)
    colors = ["#9aa5b1", "#2f6db5", "#e07b39"]
    fig, ax = plt.subplots(figsize=(10, 4.2))
    for n, row in enumerate(results.to_dict("records")):
        vals = [row[m] for m in metrics]
        bars = ax.bar(x + (n - (len(results) - 1) / 2) * w, vals, w, label=PRETTY[row["model"]], color=colors[n % 3])
        ax.bar_label(bars, fmt="%.3f", fontsize=7, padding=2)
    ax.set_xticks(x, labels)
    ax.set_ylabel("score")
    ax.set_title(f"Model comparison on the time-based test split (K = {k})")
    ax.legend(fontsize=8, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run_evaluation(cfg: Dict) -> Dict:
    """Evaluate all trained models and write the comparison artifacts."""
    proc, rep, figs, mdir = (cfg["paths"][p] for p in ("processed", "reports", "figures", "models"))
    k, rel = cfg["evaluation"]["k"], cfg["evaluation"]["relevance_threshold"]
    test = load_table(proc / "test")
    train = load_table(proc / "train")
    store = FeatureStore(cfg)
    items = store.item
    item_genres = {int(i): frozenset(g for g in GENRES if row[f"item_genre_{g}"] == 1)
                   for i, row in items[[f"item_genre_{g}" for g in GENRES]].iterrows()}
    item_share = items["item_user_share"].to_dict()
    titles = load_table(proc / "items_clean").set_index("item_id")["title"].to_dict()

    relevant = test[test.rating >= rel].groupby("user_id")["item_id"].apply(set).to_dict()
    train_users = set(train.user_id)
    warm = sorted(u for u in relevant if u in train_users)
    cold = sorted(u for u in relevant if u not in train_users)
    log.info("Evaluating on %d warm users and %d cold-start users with >= 1 relevant test item", len(warm), len(cold))

    tracker = ExperimentTracker(cfg)
    rows, per_user, samples = [], [], []
    for name in MODEL_NAMES:
        model = BaseRecommender.load(mdir / f"{name}.pkl")
        recs = {u: model.recommend(u, k) for u in warm}
        rr = ranking_report(recs, relevant, k)
        cold_rr = ranking_report({u: model.recommend(u, k) for u in cold}, relevant, k) if cold else {}
        res = {"model": name, **{m: rr[m] for m in rr if m != "n_users"}, "users_evaluated": rr["n_users"],
               "coverage": catalog_coverage(recs.values(), len(model.item_ids)),
               "diversity": float(np.mean([intra_list_diversity(r, item_genres) for r in recs.values()])),
               "novelty": float(np.mean([novelty(r, item_share) for r in recs.values()])),
               "rmse": np.nan, "mae": np.nan,
               f"cold_start_ndcg_at_{k}": cold_rr.get(f"ndcg_at_{k}", np.nan), "cold_start_users": len(cold)}
        preds = model.predict_rating(test.user_id.to_numpy(), test.item_id.to_numpy())
        if preds is not None:
            res["rmse"], res["mae"] = rmse(test.rating.to_numpy(), preds), mae(test.rating.to_numpy(), preds)
        rows.append(res)
        for u in warm:
            per_user.append({"model": name, "user_id": u, f"precision_at_{k}": precision_at_k(recs[u], relevant[u], k),
                             f"recall_at_{k}": recall_at_k(recs[u], relevant[u], k),
                             f"ndcg_at_{k}": ndcg_at_k(recs[u], relevant[u], k)})
        for u in warm[:3]:
            for rank, it in enumerate(recs[u], 1):
                samples.append({"model": name, "user_id": u, "rank": rank, "item_id": it, "title": titles.get(it, ""),
                                "relevant_in_test": int(it in relevant[u])})
        # append test metrics to the model's training run (same MLflow run id)
        run_info = load_json(mdir / f"{name}_run.json")
        with tracker.start_run(f"{name}-eval", run_id=run_info["run_id"]):
            tracker.log_metrics({f"test_{m}": v for m, v in res.items() if m != "model" and isinstance(v, (int, float))})
        log.info("%-22s P@%d=%.4f R@%d=%.4f NDCG@%d=%.4f Cov=%.3f Div=%.3f RMSE=%s", name, k, res[f"precision_at_{k}"],
                 k, res[f"recall_at_{k}"], k, res[f"ndcg_at_{k}"], res["coverage"], res["diversity"],
                 "n/a" if np.isnan(res["rmse"]) else f"{res['rmse']:.4f}")

    results = pd.DataFrame(rows)
    results.to_csv(rep / "model_comparison.csv", index=False)
    pd.DataFrame(per_user).to_csv(rep / "per_user_metrics.csv", index=False)
    pd.DataFrame(samples).to_csv(rep / "sample_recommendations.csv", index=False)
    _plot(results, k, figs / "model_comparison.png")
    _write_markdown(results, k, rel, len(warm), len(cold), rep / "model_comparison.md")
    best = results.sort_values(f"ndcg_at_{k}", ascending=False).iloc[0]["model"]
    summary = {"k": k, "warm_users": len(warm), "cold_users": len(cold), "best_model_by_ndcg": best,
               "results": results.replace({np.nan: None}).to_dict("records")}
    save_json(summary, rep / "evaluation_summary.json")
    return summary


def _write_markdown(results: pd.DataFrame, k: int, rel: int, n_warm: int, n_cold: int, path) -> None:
    """Results table in Markdown (also embedded in the final report)."""
    fmt = lambda v: "n/a" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.4f}"
    lines = ["# Model Evaluation Results", "",
             f"Time-based test split; K = {k}; relevant = test rating >= {rel}. Ranking metrics averaged over "
             f"**{n_warm} warm users** (users with training history and at least one relevant test item). "
             f"{n_cold} cold-start users are served by the popularity fallback (column *Cold NDCG*).", "",
             f"| Model | Precision@{k} | Recall@{k} | NDCG@{k} | HitRate@{k} | Coverage | Diversity | Novelty | RMSE | MAE | Cold NDCG@{k} |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in results.to_dict("records"):
        lines.append(f"| {PRETTY[r['model']]} | {fmt(r[f'precision_at_{k}'])} | {fmt(r[f'recall_at_{k}'])} | "
                     f"{fmt(r[f'ndcg_at_{k}'])} | {fmt(r[f'hit_rate_at_{k}'])} | {fmt(r['coverage'])} | "
                     f"{fmt(r['diversity'])} | {r['novelty']:.2f} | {fmt(r['rmse'])} | {fmt(r['mae'])} | "
                     f"{fmt(r[f'cold_start_ndcg_at_{k}'])} |")
    lines += ["", "*Coverage* = share of the catalogue recommended to at least one user; *Diversity* = mean pairwise "
              "genre dissimilarity (1 - Jaccard) within a top-K list; *Novelty* = mean -log2(share of users who rated "
              "the item). RMSE/MAE are computed on all test ratings for models that predict explicit ratings "
              "(popularity predicts the Bayesian item average); the hybrid model is ranking-only.", ""]
    path.write_text("\n".join(lines), encoding="utf-8")
