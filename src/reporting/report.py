"""Final-report generator (last task of the flow).

Builds ``reports/final_report.md`` and a self-contained ``reports/final_report.html``
(images embedded) from the artifacts of the CURRENT run, so every number in the
report always matches the latest pipeline execution. Open the HTML file in a
browser and use *Print -> Save as PDF* to obtain the PDF for submission.
"""
from __future__ import annotations

import base64
import re
import time
from pathlib import Path
from typing import Dict

import pandas as pd

from src.reporting.architecture import draw_architecture
from src.utils.io import load_json
from src.utils.logger import get_logger

log = get_logger(__name__)


def _pct(a: float, b: float) -> str:
    """Relative change of a vs b as a signed percentage string."""
    return f"{(a - b) / b * 100:+.1f}%" if b else "n/a"


def _interpretation(res: pd.DataFrame, k: int) -> str:
    """Data-driven discussion of the results table."""
    r = res.set_index("model")
    nd, pr = f"ndcg_at_{k}", f"precision_at_{k}"
    pop, mf, hy = r.loc["popularity"], r.loc["matrix_factorization"], r.loc["hybrid"]
    best = r[nd].idxmax()
    names = {"popularity": "the popularity baseline", "matrix_factorization": "matrix factorization", "hybrid": "the hybrid model"}
    txt = [f"* **Ranking quality.** The best model by NDCG@{k} is **{names[best]}** (NDCG@{k} = {r.loc[best, nd]:.4f}). "
           f"The hybrid model changes NDCG@{k} by {_pct(hy[nd], pop[nd])} and Precision@{k} by {_pct(hy[pr], pop[pr])} "
           f"relative to the popularity baseline."]
    if mf[nd] < pop[nd]:
        txt.append(f"* **Matrix factorization alone ranks poorly** (NDCG@{k} = {mf[nd]:.4f}) although it is the most accurate "
                   f"rating predictor (RMSE {mf['rmse']:.4f} vs {pop['rmse']:.4f} for the baseline). Minimising squared rating "
                   "error does not optimise top-K ranking: MF promotes items with high predicted ratings that few users "
                   "actually choose to watch. This is why the hybrid blends the MF score with popularity and genre affinity.")
    else:
        txt.append(f"* **Matrix factorization** improves ranking over the baseline (NDCG@{k} {mf[nd]:.4f} vs {pop[nd]:.4f}) "
                   f"and is also the most accurate rating predictor (RMSE {mf['rmse']:.4f} vs {pop['rmse']:.4f}).")
    txt.append(f"* **Beyond accuracy.** Coverage is {pop['coverage']:.1%} (popularity), {mf['coverage']:.1%} (MF) and "
               f"{hy['coverage']:.1%} (hybrid); diversity is {pop['diversity']:.3f} / {mf['diversity']:.3f} / {hy['diversity']:.3f}. "
               "The baseline recommends the same few blockbusters to everyone (lowest novelty), while the personalised models "
               "trade some of that safety for more varied and novel lists.")
    txt.append("* **Cold-start users** (no training history) receive popularity recommendations from every model, so their "
               f"NDCG@{k} is identical across models ({pop[f'cold_start_ndcg_at_{k}']:.4f}).")
    return "\n".join(txt)


def build_markdown(cfg: Dict) -> str:
    """Assemble the report text from the run's JSON/CSV artifacts."""
    P = cfg["paths"]
    k = cfg["evaluation"]["k"]
    ing = load_json(P["raw"] / "ingestion_summary.json")
    val = load_json(P["reports"] / "validation_summary.json")
    split = load_json(P["processed"] / "split_metadata.json")
    tr = load_json(P["processed"] / "transformation_summary.json")
    feat = load_json(P["features"] / "feature_metadata.json")
    hyb = load_json(P["models"] / "hybrid_run.json")
    mfp = cfg["models"]["matrix_factorization"]
    res = pd.read_csv(P["reports"] / "model_comparison.csv")
    checks = pd.read_csv(P["reports"] / "validation_checks.csv")
    comp_md = (P["reports"] / "model_comparison.md").read_text(encoding="utf-8").split("\n", 2)[2]
    c = val["counts"]
    failed = checks[checks.status == "FAIL"]
    by_cat = failed.groupby("category")["failed_rows"].sum().to_dict()

    return f"""# End-to-End Data Management Pipeline for a Recommendation System
*Data Management for Machine Learning - Assignment 2 - final report (auto-generated {time.strftime('%Y-%m-%d %H:%M')})*

## 1. Objective and data
We built a modular, reproducible pipeline that ingests user, item and interaction data, validates and prepares it,
engineers reusable features, trains and evaluates recommenders, and runs end-to-end as an orchestrated DAG.
**Data source:** {ing['source']} (MovieLens 100K format: users with age/gender/occupation, movies with release date and 19 genre
flags, and explicit 1-5 ratings with timestamps). Raw input: {ing['n_users_raw']:,} user rows, {ing['n_items_raw']:,} item rows and
{ing['n_interactions_raw']:,} interactions, including **{ing['n_injected_issues']} documented, deliberately injected data-quality
issues** (`data/raw/injected_issues.csv`) that exercise every validation rule.

![Architecture](figures/architecture_diagram.png)

## 2. Design decisions and tools
| Stage | Implementation | Why |
|---|---|---|
| Ingestion | `src/ingestion` - download MovieLens 100K, synthetic MovieLens-format fallback, raw zone kept as text | immutable raw copy; pipeline runs even offline |
| Validation | `src/validation` - rule engine ({val['checks_total']} checks: missing, type, range, duplicate, timestamp, referential) + Pandera schemas | invalid rows are separated into `data/errors/` with reasons; quality gate stops the run if > {cfg['validation']['max_error_rate']:.0%} of interactions are invalid |
| Preparation | `src/preparation` - standardise, impute optional fields, de-duplicate, **global time-based split** ({1 - cfg['preparation']['test_fraction']:.0%}/{cfg['preparation']['test_fraction']:.0%}) | mimics production (train on past, test on future); no look-ahead leakage |
| Transformation | `src/transformation` - user activity summary, item popularity (Bayesian average), sparse user-item matrix | derived datasets from TRAIN only |
| Features | `src/features` - registry -> {feat['n_features']} documented features, Parquet tables, `FeatureStore` class, catalog | one definition reused by training *and* prediction (`recommend.py`), versioned (v{feat['feature_version']}, as-of {feat['as_of']}) |
| Training | `src/training` - Popularity, biased Matrix Factorization (SGD), Hybrid; MLflow tracking | baseline + collaborative + hybrid; params, metrics, artifacts and lineage tags logged |
| Evaluation | `src/evaluation` - Precision/Recall/NDCG/HitRate@{k}, Coverage, Diversity, Novelty, RMSE/MAE | accuracy and beyond-accuracy; cold-start users reported separately |
| Orchestration | `dags/pipeline_flow.py` - Prefect flow, 10 tasks, retries on ingestion, run summary; cron deployment | one command reproduces everything from `configs/config.yaml` (seed {cfg['project']['seed']}) |

## 3. Data validation and preparation results
| Table | Raw | Valid | Rejected |
|---|---:|---:|---:|
| users | {c['users']['raw']:,} | {c['users']['valid']:,} | {c['users']['rejected']} |
| items | {c['items']['raw']:,} | {c['items']['valid']:,} | {c['items']['rejected']} |
| interactions | {c['interactions']['raw']:,} | {c['interactions']['valid']:,} | {c['interactions']['rejected']} |

{val['checks_failed']} of {val['checks_total']} rules detected problems (failed rows by category: {', '.join(f'{k_}: {v}' for k_, v in by_cat.items())}).
All {val['error_rows_total']} rejected records are stored with their failed rules in `data/errors/all_errors.csv`; details are in
`reports/validation_report.md`. Preparation imputed {split['imputed_occupation']} missing occupations and
{split['imputed_release_dates']} missing release dates and removed {split['repeat_ratings_removed']} repeated ratings.
The time split at **{split['cutoff_datetime']}** gives {split['train_rows']:,} training and {split['test_rows']:,} test interactions;
{split['cold_start_test_users']} test users have no training history (cold start). The user-item matrix is
{tr['matrix_shape'][0]} x {tr['matrix_shape'][1]} with density {tr['matrix_density']:.2%}.

## 4. Models
1. **Popularity baseline** - ranks unseen items by training popularity; predicts ratings with the Bayesian item average.
2. **Matrix Factorization** - r(u,i) = mu + b_u + b_i + p_u . q_i, {mfp['n_factors']} factors, {mfp['n_epochs']} SGD epochs,
   learning rate {mfp['learning_rate']}, L2 {mfp['regularization']}; items with < {mfp['min_item_support']} training ratings are not recommended.
3. **Hybrid** - z(MF score) + w_content x z(cosine(user genre profile, item genres)) + w_pop x z(log popularity). Weights were
   grid-searched on the last {cfg['models']['hybrid']['validation_fraction']:.0%} of the *training* period (test never used):
   **w_content = {hyb['params']['w_content']}, w_pop = {hyb['params']['w_pop']}** (validation NDCG@{k} = {hyb[f'val_ndcg_at_{k}']}).

## 5. Results
{comp_md}
![Model comparison](figures/model_comparison.png)

{_interpretation(res, k)}

## 6. Reproducibility and orchestration
`python run_pipeline.py` (or `python dags/pipeline_flow.py`) executes the 10-task Prefect DAG
ingest -> validate -> prepare -> transform -> build_features -> train_popularity / train_matrix_factorization -> train_hybrid ->
evaluate -> generate_report. All parameters live in `configs/config.yaml`; seeds are fixed; every stage writes its outputs to a
dedicated folder and a JSON summary; each run's status and task durations are written to `reports/pipeline_run_summary.json`,
and all model runs are tracked in MLflow (`mlflow ui --backend-store-uri sqlite:///mlflow.db`). `dags/schedule_pipeline.py`
deploys the same flow on a daily cron schedule.

## 7. Limitations and future work
* **Offline, single split:** one global time split and offline metrics only; a rolling (multi-window) backtest and online A/B
  testing would give more reliable estimates.
* **Cold start:** new users fall back to popularity; demographic or onboarding-based models could personalise from the first visit.
* **Small, explicit-feedback dataset:** MovieLens 100K is small and dense compared with production data; at scale the
  file-based feature store would be replaced by Feast/a database, and SGD by ALS / implicit-feedback models.
* **Ranking objective:** MF optimises rating error, not ranking; a learning-to-rank or BPR objective would align training with
  the evaluation metrics.
"""


def _to_html(md: str, reports_dir: Path) -> str:
    """Convert Markdown to a self-contained HTML page (images embedded as base64)."""
    try:
        import markdown  # type: ignore
        body = markdown.markdown(md, extensions=["tables"])
    except Exception:  # pragma: no cover - fallback without the markdown package
        body = "<pre>" + md.replace("&", "&amp;").replace("<", "&lt;") + "</pre>"

    def embed(m: re.Match) -> str:
        p = reports_dir / m.group(1)
        if not p.exists():
            return m.group(0)
        return f'src="data:image/png;base64,{base64.b64encode(p.read_bytes()).decode()}"'
    body = re.sub(r'src="([^"]+\.png)"', embed, body)
    css = """body{font-family:Segoe UI,Arial,sans-serif;max-width:980px;margin:24px auto;font-size:10.5pt;line-height:1.45;color:#1b1b1b}
h1{font-size:19pt;color:#1f3b63;margin-bottom:2px}h2{font-size:13.5pt;color:#1f3b63;border-bottom:1.5px solid #c9d6e8;padding-bottom:2px;margin-top:18px}
table{border-collapse:collapse;margin:8px 0;font-size:9pt}th,td{border:1px solid #b8c4d4;padding:3px 7px}th{background:#e8eef7}
img{max-width:100%;margin:6px 0}code{background:#f1f3f6;padding:0 3px;border-radius:3px;font-size:9pt}
@page{size:A4;margin:12mm}"""
    return f'<!DOCTYPE html><html><head><meta charset="utf-8"><title>Final report</title><style>{css}</style></head><body>{body}</body></html>'


def run_reporting(cfg: Dict) -> Dict:
    """Draw the architecture diagram and write the final report (Markdown + HTML)."""
    rep = cfg["paths"]["reports"]
    draw_architecture(cfg["paths"]["figures"] / "architecture_diagram.png")
    md = build_markdown(cfg)
    (rep / "final_report.md").write_text(md, encoding="utf-8")
    (rep / "final_report.html").write_text(_to_html(md, rep), encoding="utf-8")
    log.info("Final report written: reports/final_report.md and reports/final_report.html (print the HTML to PDF)")
    return {"report_md": str(rep / "final_report.md"), "report_html": str(rep / "final_report.html")}
