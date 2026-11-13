"""Draws the pipeline architecture diagram (``reports/figures/architecture_diagram.png``)."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

STAGE = "#e8f0fb"; STAGE_EDGE = "#2f6db5"
STORE = "#fff4e6"; STORE_EDGE = "#d9822b"
TOOL = "#eef7ee"; TOOL_EDGE = "#3c8d40"


def _box(ax, x, y, w, h, title, sub, face, edge, fs=9):
    """Rounded box with a bold title and a smaller subtitle."""
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                fc=face, ec=edge, lw=1.4))
    ax.text(x + w / 2, y + h * 0.66, title, ha="center", va="center", fontsize=fs, weight="bold", color="#1b2a3a")
    ax.text(x + w / 2, y + h * 0.30, sub, ha="center", va="center", fontsize=fs - 2, color="#3d4b5a", linespacing=1.2)


def _arrow(ax, x1, y1, x2, y2, color="#55657a", style="-|>", ls="-"):
    """Arrow between two points."""
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style, mutation_scale=12, color=color, lw=1.2, ls=ls))


def draw_architecture(path: Path) -> Path:
    """Render and save the architecture diagram."""
    fig, ax = plt.subplots(figsize=(15, 7.6))
    ax.set_xlim(0, 15); ax.set_ylim(0, 7.6); ax.axis("off")
    ax.text(7.5, 7.3, "End-to-End Data Management Pipeline for a Recommendation System",
            ha="center", fontsize=14, weight="bold", color="#1b2a3a")

    # orchestration band
    ax.add_patch(FancyBboxPatch((0.25, 3.2), 14.5, 3.55, boxstyle="round,pad=0.02,rounding_size=0.1",
                                fc="#fafbfd", ec="#9aa5b1", lw=1, ls="--"))
    ax.text(0.45, 6.55, "Orchestration: Prefect flow  dags/pipeline_flow.py  (DAG, retries, run summary; "
            "schedule: dags/schedule_pipeline.py)", fontsize=8.5, color="#55657a", style="italic")

    stages = [("1. Ingestion", "MovieLens 100K /\nsynthetic fallback\n+ documented dirty rows"),
              ("2. Validation", "rule engine + Pandera\nerror dataset,\nquality gate"),
              ("3. Preparation", "clean, impute,\nstandardise,\ntime-based split"),
              ("4. Transformation", "activity summary,\nitem popularity,\nuser-item matrix"),
              ("5. Features", "65 user / item /\ninteraction features\n+ catalog, FeatureStore"),
              ("6. Training", "Popularity baseline\nMatrix Factorization\nHybrid (tuned)"),
              ("7. Evaluation", "P/R/NDCG@10, Coverage,\nDiversity, Novelty,\nRMSE / MAE")]
    w, h, gap, y = 1.78, 1.55, 0.21, 4.75
    xs = [0.45 + i * (w + gap) for i in range(len(stages))]
    for x, (t, s) in zip(xs, stages):
        _box(ax, x, y, w, h, t, s, STAGE, STAGE_EDGE)
    for a, b in zip(xs[:-1], xs[1:]):
        _arrow(ax, a + w, y + h / 2, b, y + h / 2)

    # storage zones
    stores = [("data/raw/", "immutable raw CSV\n+ injected_issues.csv", 0),
              ("data/errors/", "rejected rows +\nerror_reasons", 1),
              ("data/processed/", "validated, clean,\ntrain / test, derived", 2.5),
              ("data/features/", "Parquet feature tables\n+ metadata (version)", 4),
              ("models/", "pickled models\n+ run info", 5),
              ("reports/", "validation report, catalog,\nresults, figures, final report", 6)]
    sy, sh = 3.45, 0.95
    for t, s, idx in stores:
        x = 0.45 + idx * (w + gap) if idx != 2.5 else xs[2] + (w + gap) / 2
        _box(ax, x, sy, w, sh, t, s, STORE, STORE_EDGE, fs=8.5)
    # stage -> store links
    links = [(0, 0), (1, 1), (2, 2.5), (3, 2.5), (4, 4), (5, 5), (6, 6)]
    for st, idx in links:
        sx = xs[st] + w / 2
        tx = (0.45 + idx * (w + gap) if idx != 2.5 else xs[2] + (w + gap) / 2) + w / 2
        _arrow(ax, sx, y, tx, sy + sh, color="#d9822b", ls="--")

    # tools row
    tools = [("configs/config.yaml", "single config + seed\n= reproducible runs"),
             ("Pandera + rules", "data contracts,\nerror dataset"),
             ("Pandas + SciPy", "cleaning, sparse matrix"),
             ("FeatureStore", "same features for\ntraining & prediction"),
             ("MLflow", "logged by training &\nevaluation: params,\nmetrics, artifacts"),
             ("recommend.py", "prediction CLI\n(serving)"),
             ("pytest", "unit tests for\nmetrics & validation")]
    for x, (t, s) in zip(xs, tools):
        _box(ax, x, 1.55, w, 1.1, t, s, TOOL, TOOL_EDGE, fs=8.5)
    ax.text(0.45, 2.85, "Supporting tools", fontsize=9, weight="bold", color="#3c8d40")

    ax.text(7.5, 0.85, "Legend:   blue = pipeline stage   |   orange = storage zone   |   green = tool / interface   |   "
            "solid arrow = task dependency   |   dashed = writes to", ha="center", fontsize=8.5, color="#55657a")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path
