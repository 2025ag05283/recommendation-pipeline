"""Stage 8 - Orchestration: the end-to-end pipeline as a Prefect flow (DAG).

Task graph (each arrow is a data dependency - a task receives the upstream
task's result, so Prefect draws the DAG and runs tasks in the right order):

    ingest -> validate -> prepare -> transform -> build_features
        -> train_popularity ----------------------------\\
        -> train_matrix_factorization -> train_hybrid ---> evaluate -> generate_report

* If Prefect is installed, ``python dags/pipeline_flow.py`` runs a Prefect flow
  (visible in the Prefect UI when a Prefect server is running).
* If Prefect is NOT installed, the same file runs as a plain, dependency-ordered
  Python workflow (identical tasks and order) - the pipeline never depends on it.

Every run writes ``reports/pipeline_run_summary.json`` (status + duration per task).
"""
from __future__ import annotations

import sys
import time
from functools import wraps
from pathlib import Path

# make "src" importable when this file is executed directly
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.config import load_config            # noqa: E402
from src.utils.io import save_json                  # noqa: E402
from src.utils.logger import get_logger             # noqa: E402

log = get_logger("pipeline")

try:
    from prefect import flow, task                  # type: ignore
    PREFECT_OK = True
except Exception:                                   # pragma: no cover - depends on environment
    PREFECT_OK = False

    def task(*dargs, **dkwargs):
        """No-op stand-in for prefect.task (supports @task and @task(...))."""
        if dargs and callable(dargs[0]):
            return dargs[0]
        return lambda fn: fn

    def flow(*dargs, **dkwargs):
        """No-op stand-in for prefect.flow."""
        if dargs and callable(dargs[0]):
            return dargs[0]
        return lambda fn: fn

RUN_LOG: list[dict] = []


def _ui_log(msg: str) -> None:
    """Also send a message to the Prefect run logger so it appears in the Prefect UI."""
    if PREFECT_OK:
        try:
            from prefect import get_run_logger
            get_run_logger().info(msg)
        except Exception:
            pass


def tracked(stage: str):
    """Decorator recording status and duration of each stage in RUN_LOG."""
    def deco(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            t0 = time.time()
            log.info("===== START %s =====", stage)
            try:
                out = fn(*args, **kwargs)
                RUN_LOG.append({"task": stage, "status": "SUCCESS", "seconds": round(time.time() - t0, 2)})
                log.info("===== END   %s (%.1fs) =====", stage, time.time() - t0)
                _ui_log(f"{stage} finished in {time.time() - t0:.1f}s -> {str(out)[:300]}")
                return out
            except Exception as exc:
                RUN_LOG.append({"task": stage, "status": f"FAILED: {exc}", "seconds": round(time.time() - t0, 2)})
                raise
        return wrapper
    return deco


# ----------------------------------------------------------------------- tasks
# Each task loads the config from its path (a plain string -> easy for Prefect to serialise).
# The `upstream` argument carries the previous task's summary and defines the DAG edges.

@task(name="ingest", retries=2, retry_delay_seconds=10)
@tracked("ingest")
def ingest_task(config_path: str | None = None) -> dict:
    """Download / generate raw data and store it in data/raw."""
    from src.ingestion.ingest import run_ingestion
    return run_ingestion(load_config(config_path))


@task(name="validate")
@tracked("validate")
def validate_task(config_path: str | None, upstream: dict) -> dict:
    """Run data-quality checks; split valid rows and the error dataset; quality gate."""
    from src.validation.validate import run_validation
    return run_validation(load_config(config_path))


@task(name="prepare")
@tracked("prepare")
def prepare_task(config_path: str | None, upstream: dict) -> dict:
    """Clean, impute, standardise and create the time-based train/test split."""
    from src.preparation.prepare import run_preparation
    return run_preparation(load_config(config_path))


@task(name="transform")
@tracked("transform")
def transform_task(config_path: str | None, upstream: dict) -> dict:
    """Build derived datasets (activity summary, item popularity, user-item matrix)."""
    from src.transformation.transform import run_transformation
    return run_transformation(load_config(config_path))


@task(name="build_features")
@tracked("build_features")
def features_task(config_path: str | None, upstream: dict) -> dict:
    """Build user/item/interaction features, store them and write the feature catalog."""
    from src.features.build_features import run_feature_engineering
    return run_feature_engineering(load_config(config_path))


@task(name="train_popularity")
@tracked("train_popularity")
def train_popularity_task(config_path: str | None, upstream: dict) -> dict:
    """Train + log the popularity baseline."""
    from src.training.train import train_popularity
    return train_popularity(load_config(config_path))


@task(name="train_matrix_factorization")
@tracked("train_matrix_factorization")
def train_mf_task(config_path: str | None, upstream: dict) -> dict:
    """Train + log the matrix-factorisation model."""
    from src.training.train import train_matrix_factorization
    return train_matrix_factorization(load_config(config_path))


@task(name="train_hybrid")
@tracked("train_hybrid")
def train_hybrid_task(config_path: str | None, upstream: dict) -> dict:
    """Tune + train + log the hybrid model (depends on the MF model)."""
    from src.training.train import train_hybrid
    return train_hybrid(load_config(config_path))


@task(name="evaluate")
@tracked("evaluate")
def evaluate_task(config_path: str | None, upstream_models: list) -> dict:
    """Evaluate all models on the test split and write the comparison table."""
    from src.evaluation.evaluate import run_evaluation
    return run_evaluation(load_config(config_path))


@task(name="generate_report")
@tracked("generate_report")
def report_task(config_path: str | None, upstream: dict) -> dict:
    """Render the architecture diagram and the final report (Markdown + HTML)."""
    from src.reporting.report import run_reporting
    return run_reporting(load_config(config_path))


# ------------------------------------------------------------------------ flow
@flow(name="recommendation-pipeline", log_prints=True)
def recommendation_pipeline(config_path: str | None = None) -> dict:
    """End-to-end recommendation data pipeline (see module docstring for the DAG)."""
    RUN_LOG.clear()
    t0 = time.time()
    cfg = load_config(config_path)
    status = "SUCCESS"
    try:
        raw = ingest_task(config_path)
        valid = validate_task(config_path, raw)
        prepared = prepare_task(config_path, valid)
        transformed = transform_task(config_path, prepared)
        feats = features_task(config_path, transformed)
        pop = train_popularity_task(config_path, feats)
        mf = train_mf_task(config_path, feats)
        hyb = train_hybrid_task(config_path, mf)
        ev = evaluate_task(config_path, [pop, mf, hyb])
        report_task(config_path, ev)
        return ev
    except Exception:
        status = "FAILED"
        raise
    finally:
        summary = {"flow": "recommendation-pipeline", "orchestrator": "prefect" if PREFECT_OK else "python (fallback)",
                   "status": status, "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                   "total_seconds": round(time.time() - t0, 1), "tasks": RUN_LOG}
        save_json(summary, cfg["paths"]["reports"] / "pipeline_run_summary.json")
        log.info("Pipeline %s in %.1fs (orchestrator: %s)", status, summary["total_seconds"], summary["orchestrator"])


if __name__ == "__main__":
    recommendation_pipeline()
