"""Single entry point: run the whole pipeline, or one stage.

Examples
--------
    python run_pipeline.py                     # full pipeline (Prefect flow if Prefect is installed)
    python run_pipeline.py --stage validate    # re-run a single stage (upstream outputs must exist)
    python run_pipeline.py --config configs/config.yaml
"""
from __future__ import annotations

import argparse

from dags.pipeline_flow import recommendation_pipeline
from src.utils.config import load_config

STAGES = ["ingest", "validate", "prepare", "transform", "features", "train", "evaluate", "report"]


def run_stage(stage: str, config_path: str | None) -> None:
    """Run one stage directly (plain Python, no orchestrator)."""
    cfg = load_config(config_path)
    if stage == "ingest":
        from src.ingestion.ingest import run_ingestion as fn
    elif stage == "validate":
        from src.validation.validate import run_validation as fn
    elif stage == "prepare":
        from src.preparation.prepare import run_preparation as fn
    elif stage == "transform":
        from src.transformation.transform import run_transformation as fn
    elif stage == "features":
        from src.features.build_features import run_feature_engineering as fn
    elif stage == "train":
        from src.training.train import run_training as fn
    elif stage == "evaluate":
        from src.evaluation.evaluate import run_evaluation as fn
    else:
        from src.reporting.report import run_reporting as fn
    fn(cfg)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="End-to-end recommendation data pipeline")
    parser.add_argument("--stage", choices=["all"] + STAGES, default="all", help="stage to run (default: all)")
    parser.add_argument("--config", default=None, help="path to an alternative config.yaml")
    args = parser.parse_args()
    if args.stage == "all":
        recommendation_pipeline(args.config)
    else:
        run_stage(args.stage, args.config)
