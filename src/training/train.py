"""Stage 6 - Model training (with experiment tracking).

Each model has its own entry point so the orchestrator can run them as
separate tasks (popularity and MF in parallel, hybrid after MF):

* :func:`train_popularity`
* :func:`train_matrix_factorization`
* :func:`train_hybrid` - tunes its blend weights on a time-based validation
  split taken from the END of the training data (test data is never used).

Every run logs parameters, training metrics, tags and the pickled model to
MLflow; the run id is stored in ``models/<model>_run.json`` so the evaluation
stage can append test metrics to the same run.
"""
from __future__ import annotations

import itertools
import time
from typing import Dict

import numpy as np
import pandas as pd

from src.evaluation.metrics import ranking_report
from src.features.build_features import compute_genre_preferences
from src.features.store import FeatureStore
from src.training.models import (BaseRecommender, HybridRecommender, MatrixFactorizationRecommender,
                                 PopularityRecommender)
from src.utils.io import load_json, load_table, save_json
from src.utils.logger import get_logger
from src.utils.tracking import ExperimentTracker

log = get_logger(__name__)


def _load_train(cfg: Dict) -> pd.DataFrame:
    """Load the training split."""
    train = load_table(cfg["paths"]["processed"] / "train")
    train["datetime"] = pd.to_datetime(train["datetime"])
    return train


def _finish(cfg: Dict, model: BaseRecommender, tracker: ExperimentTracker, run_id: str, extra: Dict) -> Dict:
    """Persist the model, log it as an artifact and write the run-info file."""
    path = model.save(cfg["paths"]["models"] / f"{model.name}.pkl")
    tracker.log_artifact(path)
    info = {"model": model.name, "run_id": run_id, "model_path": str(path),
            "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"), "fit_seconds": round(model.fit_seconds, 2),
            "params": model.params(), **extra}
    save_json(info, cfg["paths"]["models"] / f"{model.name}_run.json")
    log.info("Trained %-22s in %6.1fs -> %s", model.name, model.fit_seconds, path.name)
    return info


def _common_tags(cfg: Dict, train: pd.DataFrame) -> Dict:
    """Tags that describe the data/feature versions behind a run (lineage)."""
    meta = load_json(cfg["paths"]["features"] / "feature_metadata.json")
    split = load_json(cfg["paths"]["processed"] / "split_metadata.json")
    src = load_json(cfg["paths"]["raw"] / "ingestion_summary.json")["source"]
    return {"data_source": src, "feature_version": meta["feature_version"], "train_cutoff": split["cutoff_datetime"],
            "train_rows": len(train), "pipeline": cfg["project"]["name"]}


# -------------------------------------------------------------------- models
def train_popularity(cfg: Dict) -> Dict:
    """Fit and log the popularity baseline."""
    train, store, tracker = _load_train(cfg), FeatureStore(cfg), ExperimentTracker(cfg)
    with tracker.start_run("popularity") as run_id:
        model = PopularityRecommender().fit(train, store)
        tracker.set_tags(_common_tags(cfg, train) | {"model_type": "baseline"})
        tracker.log_params(model.params())
        tracker.log_metrics({"fit_seconds": model.fit_seconds, "n_items_ranked": float(len(model.item_ids))})
        return _finish(cfg, model, tracker, run_id, {})


def train_matrix_factorization(cfg: Dict) -> Dict:
    """Fit and log the biased matrix-factorisation model."""
    train, store, tracker = _load_train(cfg), FeatureStore(cfg), ExperimentTracker(cfg)
    p = cfg["models"]["matrix_factorization"]
    with tracker.start_run("matrix_factorization") as run_id:
        model = MatrixFactorizationRecommender(**p, seed=cfg["project"]["seed"]).fit(train, store)
        tracker.set_tags(_common_tags(cfg, train) | {"model_type": "personalized/collaborative"})
        tracker.log_params(model.params())
        for epoch, val in enumerate(model.train_rmse_history, 1):        # learning curve
            if tracker.use_mlflow:
                import mlflow
                mlflow.log_metric("train_rmse_epoch", val, step=epoch)
        tracker.log_metrics({"train_rmse": model.train_rmse_history[-1], "fit_seconds": model.fit_seconds})
        return _finish(cfg, model, tracker, run_id, {"train_rmse": model.train_rmse_history[-1]})


def _tune_hybrid(cfg: Dict, train: pd.DataFrame, store: FeatureStore) -> tuple[pd.DataFrame, Dict]:
    """Grid-search hybrid weights on a time-based validation split cut from the end of TRAIN."""
    hcfg, k, rel = cfg["models"]["hybrid"], cfg["evaluation"]["k"], cfg["evaluation"]["relevance_threshold"]
    cut = train["timestamp"].quantile(1 - hcfg["validation_fraction"])
    inner, val = train[train.timestamp <= cut], train[train.timestamp > cut]
    log.info("Hybrid tuning: inner-train %d rows, validation %d rows (cut %s)", len(inner), len(val),
             pd.to_datetime(int(cut), unit="s"))
    mf = MatrixFactorizationRecommender(**cfg["models"]["matrix_factorization"], seed=cfg["project"]["seed"]).fit(inner, store)
    items = load_table(cfg["paths"]["processed"] / "items_clean")
    prefs, _ = compute_genre_preferences(inner, items, rel)                 # same feature code as the feature store
    relevant = val[val.rating >= rel].groupby("user_id")["item_id"].apply(set).to_dict()
    users = [u for u in relevant if mf.is_known_user(u)]
    rows = []
    for wc, wp in itertools.product(hcfg["content_weight_grid"], hcfg["popularity_weight_grid"]):
        h = HybridRecommender(mf, wc, wp).fit(inner, store, user_prefs=prefs)
        rep = ranking_report({u: h.recommend(u, k) for u in users}, relevant, k)
        rows.append({"w_content": wc, "w_pop": wp, **rep})
    grid = pd.DataFrame(rows).sort_values(f"ndcg_at_{k}", ascending=False).reset_index(drop=True)
    return grid, grid.iloc[0].to_dict()


def train_hybrid(cfg: Dict) -> Dict:
    """Tune and fit the hybrid model on top of the trained MF model, and log it."""
    train, store, tracker = _load_train(cfg), FeatureStore(cfg), ExperimentTracker(cfg)
    k = cfg["evaluation"]["k"]
    mf = BaseRecommender.load(cfg["paths"]["models"] / "matrix_factorization.pkl")
    with tracker.start_run("hybrid") as run_id:
        t0 = time.time()
        grid, best = _tune_hybrid(cfg, train, store)
        grid_path = cfg["paths"]["reports"] / "hybrid_tuning_grid.csv"
        grid.to_csv(grid_path, index=False)
        log.info("Hybrid tuning grid (validation NDCG@%d):\n%s", k, grid.round(4).to_string(index=False))
        model = HybridRecommender(mf, best["w_content"], best["w_pop"]).fit(train, store)
        model.fit_seconds = time.time() - t0 + mf.fit_seconds
        tracker.set_tags(_common_tags(cfg, train) | {"model_type": "personalized/hybrid (CF + content + popularity)"})
        tracker.log_params(model.params() | {"tuning_grid_size": len(grid),
                                             "validation_fraction": cfg["models"]["hybrid"]["validation_fraction"]})
        tracker.log_metrics({f"val_ndcg_at_{k}": best[f"ndcg_at_{k}"], f"val_precision_at_{k}": best[f"precision_at_{k}"],
                             "fit_seconds": model.fit_seconds})
        tracker.log_artifact(grid_path)
        return _finish(cfg, model, tracker, run_id, {f"val_ndcg_at_{k}": round(best[f"ndcg_at_{k}"], 4)})


def run_training(cfg: Dict) -> Dict:
    """Train all models sequentially (used by the plain-Python runner)."""
    return {"popularity": train_popularity(cfg), "matrix_factorization": train_matrix_factorization(cfg),
            "hybrid": train_hybrid(cfg)}
