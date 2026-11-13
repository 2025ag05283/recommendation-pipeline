"""Build the comprehensive SUBMISSION report (HTML -> print to PDF).

Reads the artifacts of YOUR latest pipeline run (JSON summaries, CSV/Parquet
tables, figures) and every screenshot in ``reports/figures/`` and assembles a
single self-contained HTML file: ``reports/submission_report.html``.

Screenshot naming (put the PNG files in reports/figures/):
    stage1_*.png  ingestion        stage5_*.png  features
    stage2_*.png  validation       stage6_*.png  training
    stage3_*.png  preparation      stage7_*.png  evaluation
    stage4_*.png  transformation   stage8_*.png  report generation
    01_*.png .. 03_*.png  orchestration (terminal success, Prefect graph, Prefect task runs)
    04_*.png, 05_*.png    MLflow       06_*.png  prediction demo
Several screenshots per stage are allowed (e.g. stage2_a_terminal.png, stage2_b_report.png).
A missing screenshot is shown as a placeholder box that tells you which command to run.

Usage:
    python scripts/build_submission_report.py --group "Individual submission" \
        --member "Full Name (ID)" \
        --github https://github.com/<user>/recommendation-pipeline
Then open reports/submission_report.html in Chrome -> Ctrl+P -> Save as PDF.
"""
from __future__ import annotations

import argparse
import base64
import html
import json
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.reporting.report import _interpretation  # noqa: E402
from src.utils.config import load_config            # noqa: E402
from src.utils.io import load_json, load_table      # noqa: E402

PRETTY = {"popularity": "Popularity (baseline)", "matrix_factorization": "Matrix Factorization (SVD)",
          "hybrid": "Hybrid (MF + content + popularity)"}


# ------------------------------------------------------------------- helpers
def esc(x) -> str:
    """HTML-escape any value."""
    return html.escape(str(x))


def img_tag(path: Path, caption: str) -> str:
    """Embed a PNG as base64 inside a captioned figure."""
    data = base64.b64encode(path.read_bytes()).decode()
    return (f'<figure><img src="data:image/png;base64,{data}" alt="{esc(caption)}"/>'
            f'<figcaption>{esc(caption)}</figcaption></figure>')


def table(df: pd.DataFrame, cls: str = "") -> str:
    """Render a DataFrame as a compact HTML table."""
    return df.to_html(index=False, border=0, classes=f"tbl {cls}", escape=True, na_rep="-", float_format=lambda v: f"{v:,.4f}")


def kv(d: dict) -> str:
    """Render key/value pairs as a two-column table."""
    rows = "".join(f"<tr><th>{esc(k)}</th><td>{esc(v)}</td></tr>" for k, v in d.items())
    return f'<table class="tbl kv">{rows}</table>'


def code(cmd: str) -> str:
    """Command block."""
    return f"<pre class='cmd'>{esc(cmd)}</pre>"


class Shots:
    """Finds screenshots in reports/figures and keeps a running figure counter."""

    def __init__(self, folder: Path):
        self.folder, self.n = folder, 0

    def section(self, patterns: list[str], label: str, command: str) -> str:
        """All screenshots matching the patterns, or a placeholder telling the user what to capture."""
        files = []
        for p in patterns:
            files += sorted(self.folder.glob(p))
        if not files:
            return (f"<div class='placeholder'><b>Execution screenshot - {esc(label)}</b><br>"
                    f"Run <code>{esc(command)}</code>, screenshot the terminal and save it as "
                    f"<code>reports/figures/{esc(patterns[0].replace('*', '_terminal'))}</code>, "
                    "then rebuild this report.</div>")
        out = []
        for f in files:
            self.n += 1
            nice = f.stem.split("_", 1)[-1].replace("_", " ")
            out.append(img_tag(f, f"Figure {self.n}. {label} - {nice}"))
        return "".join(out)


# ------------------------------------------------------------------- builder
def build(cfg: dict, args) -> str:
    P = cfg["paths"]
    k = cfg["evaluation"]["k"]
    shots = Shots(P["figures"])
    ing = load_json(P["raw"] / "ingestion_summary.json")
    val = load_json(P["reports"] / "validation_summary.json")
    split = load_json(P["processed"] / "split_metadata.json")
    tr = load_json(P["processed"] / "transformation_summary.json")
    feat = load_json(P["features"] / "feature_metadata.json")
    run_sum = load_json(P["reports"] / "pipeline_run_summary.json") if (P["reports"] / "pipeline_run_summary.json").exists() else {}
    res = pd.read_csv(P["reports"] / "model_comparison.csv")
    checks = pd.read_csv(P["reports"] / "validation_checks.csv")
    catalog = pd.read_csv(P["reports"] / "feature_catalog.csv")
    items = load_table(P["processed"] / "items_clean")
    titles = items.set_index("item_id")["title"].to_dict()
    runs = {m: load_json(P["models"] / f"{m}_run.json") for m in PRETTY}
    best = res.sort_values(f"ndcg_at_{k}", ascending=False).iloc[0]
    base = res.set_index("model").loc["popularity"]
    lift = (best[f"ndcg_at_{k}"] / base[f"ndcg_at_{k}"] - 1) * 100

    S = []  # html sections

    # ---------------------------------------------------------------- cover
    members = "".join(f"<li>{esc(m)}</li>" for m in args.member) or "<li><i>add with --member \"Name (ID)\"</i></li>"
    status = run_sum.get("status", "SUCCESS")
    S.append(f"""
<section class="cover">
  <div class="course">Data Management for Machine Learning (DM4ML) - Assignment 2</div>
  <h1>End-to-End Data Management Pipeline<br>for a Recommendation System</h1>
  <div class="sub">MovieLens 100K - Ingestion - Validation - Preparation - Transformation - Features - Training - Evaluation - Orchestration</div>
  <table class="tbl kv cover-kv">
    <tr><th>Submission</th><td>{esc(args.group)}</td></tr>
    <tr><th>{'Student' if len(args.member) <= 1 else 'Members'}</th><td><ul>{members}</ul></td></tr>
    <tr><th>GitHub repository</th><td><a href="{esc(args.github)}">{esc(args.github)}</a></td></tr>
    <tr><th>Pipeline run</th><td>{esc(status)} - {esc(run_sum.get('finished_at', '-'))} - {esc(run_sum.get('total_seconds', '-'))} s -
        orchestrator: {esc(run_sum.get('orchestrator', '-'))}</td></tr>
    <tr><th>Report generated</th><td>{time.strftime('%Y-%m-%d %H:%M')}</td></tr>
  </table>
  <div class="kpis">
    <div><b>{ing['n_interactions_raw']:,}</b><span>raw interactions</span></div>
    <div><b>{val['counts']['interactions']['rejected'] + val['counts']['users']['rejected'] + val['counts']['items']['rejected']}</b><span>invalid records caught</span></div>
    <div><b>{feat['n_features']}</b><span>documented features</span></div>
    <div><b>3</b><span>models tracked in MLflow</span></div>
    <div><b>{best[f'ndcg_at_{k}']:.4f}</b><span>best NDCG@{k} ({esc(PRETTY[best['model']].split(' (')[0])})</span></div>
    <div><b>{lift:+.1f}%</b><span>NDCG lift vs baseline</span></div>
  </div>
  <h3>Contents</h3>
  <ol class="toc">
    <li>Overview, architecture and how to run</li><li>Stage 1 - Data ingestion</li><li>Stage 2 - Data validation</li>
    <li>Stage 3 - Data preparation</li><li>Stage 4 - Data transformation</li><li>Stage 5 - Feature management</li>
    <li>Stage 6 - Model training and experiment tracking</li><li>Stage 7 - Model evaluation</li>
    <li>Stage 8 - Orchestration and reproducibility</li><li>Prediction using the feature store</li>
    <li>Conclusions and limitations</li><li>Appendix - full feature catalog</li>
  </ol>
</section>""")

    # ---------------------------------------------------------------- 1 overview
    arch = P["figures"] / "architecture_diagram.png"
    S.append(f"""
<section><h2>1. Overview, architecture and how to run</h2>
<p>The pipeline turns raw user, item and interaction data into evaluated recommendation models. It is implemented as
independent Python modules (one per stage) that communicate only through files in dedicated storage zones
(<code>data/raw</code> &rarr; <code>data/processed</code> &rarr; <code>data/features</code> &rarr; <code>models</code> &rarr; <code>reports</code>).
All parameters are in <code>configs/config.yaml</code> (seed {cfg['project']['seed']}), and the stages are chained as a Prefect DAG.</p>
{img_tag(arch, 'Figure 0. Pipeline architecture: stages (blue), storage zones (orange), tools (green)') if arch.exists() else ''}
<h3>Tools and design decisions</h3>
<table class="tbl">
<tr><th>Concern</th><th>Choice</th><th>Reason</th></tr>
<tr><td>Dataset</td><td>MovieLens 100K (GroupLens)</td><td>users, items and timestamped explicit ratings, as required by the brief</td></tr>
<tr><td>Validation</td><td>Custom rule engine ({val['checks_total']} rules) + Pandera schemas</td><td>row-level error dataset with reasons, plus a declarative data contract</td></tr>
<tr><td>Storage</td><td>Parquet (columnar, typed) + JSON metadata</td><td>fast, schema-preserving, versionable</td></tr>
<tr><td>Feature store</td><td>File-based <code>FeatureStore</code> + registry + catalog</td><td>one definition reused at training and prediction time</td></tr>
<tr><td>Models</td><td>Popularity, Matrix Factorization (SGD), Hybrid</td><td>baseline vs collaborative vs hybrid comparison</td></tr>
<tr><td>Tracking</td><td>MLflow {esc('(SQLite backend)')}</td><td>parameters, metrics, artifacts and lineage tags per run</td></tr>
<tr><td>Orchestration</td><td>Prefect flow with 10 tasks</td><td>explicit DAG, retries, UI evidence, cron scheduling</td></tr>
</table>
<h3>How to reproduce</h3>
{code('python -m venv .venv  &&  .venv\\Scripts\\activate\npython -m pip install -r requirements.txt\nprefect server start                       # terminal 1\nprefect config set PREFECT_API_URL=http://127.0.0.1:4200/api\npython run_pipeline.py                     # terminal 2: full DAG\nmlflow ui --backend-store-uri sqlite:///mlflow.db\npython recommend.py --user 42')}
</section>""")

    # ---------------------------------------------------------------- 2 ingestion
    inj_file = P["raw"] / "injected_issues.csv"
    inj = pd.read_csv(inj_file) if inj_file.exists() else pd.DataFrame(columns=["table", "issue"])
    inj_sum = inj.groupby(["table", "issue"]).size().reset_index(name="records") if len(inj) else inj
    raw_x = pd.read_csv(P["raw"] / "interactions.csv", dtype=str).head(6)
    S.append(f"""
<section><h2>2. Stage 1 - Data ingestion</h2>
<p><b>Module:</b> <code>src/ingestion/ingest.py</code>. Downloads MovieLens 100K, parses <code>u.user</code>, <code>u.item</code>
and <code>u.data</code> <b>as text</b> (type checking is left to validation), and stores an immutable raw copy in <code>data/raw/</code>.
To prove that validation works, a small, fully documented set of invalid records is injected
(<code>data/raw/injected_issues.csv</code>).</p>
{kv({'Source': ing['source'], 'Raw users': f"{ing['n_users_raw']:,}", 'Raw items': f"{ing['n_items_raw']:,}",
     'Raw interactions': f"{ing['n_interactions_raw']:,}", 'Injected issues': ing['n_injected_issues'],
     'Ingested at': ing['ingested_at'], 'Duration': f"{ing['seconds']} s"})}
<h3>Injected data-quality issues (by type)</h3>{table(inj_sum)}
<h3>Sample of raw interactions (as stored, untyped)</h3>{table(raw_x)}
<h3>Execution evidence</h3>
{shots.section(['stage1_*.png'], 'Stage 1 ingestion', 'python run_pipeline.py --stage ingest')}
</section>""")

    # ---------------------------------------------------------------- 3 validation
    c = val["counts"]
    cnt = pd.DataFrame([{"table": t, "raw rows": v["raw"], "valid rows": v["valid"], "rejected rows": v["rejected"],
                         "rejected %": round(v["rejected_pct"], 3)} for t, v in c.items()])
    errs = pd.read_csv(P["errors"] / "all_errors.csv").head(12)
    errs["record"] = errs["record"].astype(str).str.slice(0, 70)
    S.append(f"""
<section><h2>3. Stage 2 - Data validation</h2>
<p><b>Modules:</b> <code>src/validation/checks.py</code>, <code>schemas.py</code>, <code>validate.py</code>. Users and items are validated first,
then interactions (so referential integrity is checked against <i>valid</i> ids). Every failing row is written to
<code>data/errors/</code> with the list of rules it broke; valid rows are typed and saved to <code>data/processed/*_validated</code>.
A quality gate stops the pipeline if more than {cfg['validation']['max_error_rate']:.0%} of interactions are invalid.
Pandera available: <b>{val['pandera_available']}</b>.</p>
<h3>Row counts</h3>{table(cnt)}
<h3>Validation rules and results ({val['checks_failed']} of {val['checks_total']} rules detected problems)</h3>
{table(checks[['table', 'check', 'category', 'rule', 'failed_rows', 'status']])}
<h3>Sample of the error dataset (<code>data/errors/all_errors.csv</code>)</h3>{table(errs)}
<h3>Execution evidence</h3>
{shots.section(['stage2_*.png'], 'Stage 2 validation', 'python run_pipeline.py --stage validate')}
</section>""")

    # ---------------------------------------------------------------- 4 preparation
    train = load_table(P["processed"] / "train").head(6)
    S.append(f"""
<section><h2>4. Stage 3 - Data preparation</h2>
<p><b>Module:</b> <code>src/preparation/prepare.py</code>. Standardises formats (gender upper-case, occupation lower-case, dates parsed),
imputes missing optional values (occupation/zip &rarr; "unknown", missing release date &rarr; median date with an
<code>release_date_imputed</code> flag), keeps only the latest rating when a user rated the same item twice, and creates a
<b>global time-based split</b>: everything up to the {1 - cfg['preparation']['test_fraction']:.0%} timestamp quantile is training data, later interactions are test data.
This simulates production (train on the past, predict the future) and avoids look-ahead leakage.</p>
{kv({'Train/test cutoff': split['cutoff_datetime'], 'Training interactions': f"{split['train_rows']:,}",
     'Test interactions': f"{split['test_rows']:,}", 'Train period': ' to '.join(split['train_period']),
     'Test period': ' to '.join(split['test_period']), 'Users in test': split['users_in_test'],
     'Cold-start test users (no training history)': split['cold_start_test_users'],
     'Cold-start test items': split['cold_start_test_items'], 'Repeated ratings removed': split['repeat_ratings_removed'],
     'Occupations imputed': split['imputed_occupation'], 'Release dates imputed': split['imputed_release_dates']})}
<h3>Sample of the cleaned training split</h3>{table(train)}
<h3>Execution evidence</h3>
{shots.section(['stage3_*.png'], 'Stage 3 preparation', 'python run_pipeline.py --stage prepare')}
</section>""")

    # ---------------------------------------------------------------- 5 transformation
    pop = load_table(P["processed"] / "item_popularity").head(10)
    pop.insert(1, "title", pop["item_id"].map(titles))
    pop = pop[["popularity_rank", "item_id", "title", "n_ratings", "avg_rating", "bayesian_avg_rating", "user_share"]]
    uas = load_table(P["processed"] / "user_activity_summary")
    uas_desc = uas[["n_interactions", "avg_rating", "rating_std", "active_span_days"]].describe().T.reset_index().rename(columns={"index": "statistic"})
    uas_desc = uas_desc[["statistic", "mean", "std", "min", "50%", "max"]].round(3)
    S.append(f"""
<section><h2>5. Stage 4 - Data transformation</h2>
<p><b>Module:</b> <code>src/transformation/transform.py</code>. Builds derived datasets from the <b>training split only</b>:
a user activity summary, item popularity (with a Bayesian-average rating that shrinks items with few ratings towards the
global mean), a one-hot item-genre matrix, and a sparse user-item rating matrix
({tr['matrix_shape'][0]} x {tr['matrix_shape'][1]}, {tr['matrix_nnz']:,} ratings, density {tr['matrix_density']:.2%}).</p>
<h3>Top-10 items by training popularity</h3>{table(pop)}
<h3>User activity summary (distribution over {len(uas)} users)</h3>{table(uas_desc)}
<h3>Execution evidence</h3>
{shots.section(['stage4_*.png'], 'Stage 4 transformation', 'python run_pipeline.py --stage transform')}
</section>""")

    # ---------------------------------------------------------------- 6 features
    fs_counts = catalog.groupby("entity").size().reset_index(name="features")
    uf = load_table(P["features"] / "user_features")
    uf_cols = ["user_id", "user_age", "user_gender", "user_activity_count", "user_avg_rating", "user_rating_std",
               "user_preferred_genre", "user_days_since_last_interaction"]
    itf = load_table(P["features"] / "item_features")
    it_cols = ["item_id", "item_popularity", "item_avg_rating", "item_bayesian_avg_rating", "item_age_years", "item_n_genres"]
    key_feats = catalog[catalog["name"].isin(["user_avg_rating", "user_activity_count", "user_preferred_genre",
                                              "user_days_since_last_interaction", "item_popularity", "item_bayesian_avg_rating",
                                              "item_age_years", "rating_dev_from_user_avg", "interaction_recency_days"])]
    S.append(f"""
<section><h2>6. Stage 5 - Feature management</h2>
<p><b>Modules:</b> <code>src/features/registry.py</code> (metadata of every feature), <code>build_features.py</code>, <code>store.py</code>.
Features are computed from training data <b>as of the train cutoff</b> ({esc(feat['as_of'])}), stored as Parquet tables in
<code>data/features/</code> with versioned metadata (v{esc(feat['feature_version'])}), and checked against the registry (built columns
must equal declared columns). The same <code>FeatureStore</code> class serves features to model training <i>and</i> to prediction
(<code>recommend.py</code>), which avoids training/serving skew. The full catalog is in the appendix.</p>
<table class="tbl"><tr><th>Entity</th><th>Features</th><th>Rows stored</th></tr>
{''.join(f"<tr><td>{esc(r.entity)}</td><td>{r.features}</td><td>{feat['tables'][r.entity + '_features']['rows']:,}</td></tr>" for r in fs_counts.itertuples())}
</table>
<h3>Key features (excerpt of the catalog)</h3>{table(key_feats[['name', 'entity', 'dtype', 'description', 'logic']])}
<h3>Sample - user features</h3>{table(uf[uf_cols].head(5).round(3))}
<h3>Sample - item features</h3>{table(itf[it_cols].head(5).round(3))}
<h3>Execution evidence</h3>
{shots.section(['stage5_*.png'], 'Stage 5 feature engineering', 'python run_pipeline.py --stage features')}
</section>""")

    # ---------------------------------------------------------------- 7 training
    grid_f = P["reports"] / "hybrid_tuning_grid.csv"
    grid = pd.read_csv(grid_f).head(6).round(4) if grid_f.exists() else pd.DataFrame()
    prm = pd.DataFrame([{"model": PRETTY[m], "MLflow run id": r["run_id"][:12] + "...", "fit time (s)": r["fit_seconds"],
                         "parameters": ", ".join(f"{a}={b}" for a, b in r["params"].items() if a != "model")} for m, r in runs.items()])
    S.append(f"""
<section><h2>7. Stage 6 - Model training and experiment tracking</h2>
<p><b>Modules:</b> <code>src/training/models.py</code>, <code>train.py</code>. All models share one interface
(<code>fit / score / recommend / predict_rating</code>) and fall back to popularity for cold-start users.</p>
<ul>
<li><b>Popularity (baseline)</b> - recommends the most-rated unseen items; predicts ratings with the Bayesian item average.</li>
<li><b>Matrix Factorization</b> - r(u,i) = mu + b_u + b_i + p_u.q_i, trained with SGD (train RMSE {runs['matrix_factorization'].get('train_rmse', 0):.4f}).</li>
<li><b>Hybrid</b> - z(MF score) + w_content &middot; z(genre affinity from the feature store) + w_pop &middot; z(log popularity).
Weights were grid-searched on the last {cfg['models']['hybrid']['validation_fraction']:.0%} of the <i>training</i> period (the test set is never used for tuning).</li>
</ul>
<h3>Trained models (logged to MLflow with parameters, metrics, lineage tags and the model artifact)</h3>{table(prm)}
<h3>Hybrid weight tuning - best validation configurations</h3>{table(grid)}
<h3>Execution evidence</h3>
{shots.section(['stage6_*.png'], 'Stage 6 training', 'python run_pipeline.py --stage train')}
{shots.section(['04_*.png', '05_*.png'], 'MLflow experiment tracking', 'mlflow ui --backend-store-uri sqlite:///mlflow.db')}
</section>""")

    # ---------------------------------------------------------------- 8 evaluation
    shown = res.copy()
    shown["model"] = shown["model"].map(PRETTY)
    cols = ["model", f"precision_at_{k}", f"recall_at_{k}", f"ndcg_at_{k}", f"hit_rate_at_{k}", "coverage", "diversity",
            "novelty", "rmse", "mae", "users_evaluated"]
    samples = pd.read_csv(P["reports"] / "sample_recommendations.csv")
    s_user = samples[samples.model == "hybrid"]["user_id"].iloc[0]
    sample = samples[(samples.model == "hybrid") & (samples.user_id == s_user)][["rank", "item_id", "title", "relevant_in_test"]]
    chart = P["figures"] / "model_comparison.png"
    interp = _interpretation(res, k).replace("**", "")
    interp_html = "".join(f"<li>{esc(line.lstrip('* '))}</li>" for line in interp.split("\n") if line.strip())
    ev = load_json(P["reports"] / "evaluation_summary.json")
    S.append(f"""
<section><h2>8. Stage 7 - Model evaluation</h2>
<p><b>Modules:</b> <code>src/evaluation/metrics.py</code> (unit-tested), <code>evaluate.py</code>. Test period = interactions after the
cutoff; relevant item = test rating &ge; {cfg['evaluation']['relevance_threshold']}. Ranking metrics are averaged over
<b>{ev['warm_users']} warm users</b> (with training history and at least one relevant test item); {ev['cold_users']} cold-start users
receive the popularity fallback. RMSE/MAE are computed on all test ratings for models that predict ratings.</p>
<h3>Model comparison (K = {k})</h3>{table(shown[cols])}
{img_tag(chart, 'Model comparison on the time-based test split') if chart.exists() else ''}
<h3>Interpretation</h3><ul>{interp_html}</ul>
<h3>Example - hybrid top-{k} for user {s_user} (1 = item was rated &ge; {cfg['evaluation']['relevance_threshold']} in the test period)</h3>{table(sample)}
<h3>Execution evidence</h3>
{shots.section(['stage7_*.png'], 'Stage 7 evaluation', 'python run_pipeline.py --stage evaluate')}
</section>""")

    # ---------------------------------------------------------------- 9 orchestration
    tasks = pd.DataFrame(run_sum.get("tasks", []))
    S.append(f"""
<section><h2>9. Stage 8 - Orchestration and reproducibility</h2>
<p><b>Module:</b> <code>dags/pipeline_flow.py</code>. The pipeline is a Prefect flow of 10 tasks. Each task receives the previous
task's result, so Prefect draws the dependency graph:
<code>ingest &rarr; validate &rarr; prepare &rarr; transform &rarr; build_features &rarr; train_popularity / train_matrix_factorization &rarr;
train_hybrid &rarr; evaluate &rarr; generate_report</code>. Ingestion retries twice on failure (network), the validation quality gate can
stop the run, and every run writes <code>reports/pipeline_run_summary.json</code>. <code>dags/schedule_pipeline.py</code> deploys
the same flow on a daily cron schedule (02:00). One command, <code>python run_pipeline.py</code>, reproduces everything from the config.</p>
<h3>Task run summary of the reported run ({esc(run_sum.get('orchestrator', '-'))}, {esc(run_sum.get('total_seconds', '-'))} s)</h3>
{table(tasks) if len(tasks) else ''}
<h3>Execution evidence</h3>
{shots.section(['01_*.png', '02_*.png', '03_*.png'], 'Full pipeline run', 'python run_pipeline.py')}
{shots.section(['stage8_*.png'], 'Report generation', 'python run_pipeline.py --stage report')}
</section>""")

    # ---------------------------------------------------------------- 10 prediction
    S.append(f"""
<section><h2>10. Prediction using the feature store</h2>
<p><code>recommend.py</code> loads a trained model and the <i>same</i> <code>FeatureStore</code> that was used in training, prints
the user's features (activity, average rating, preferred genre, recency) and the top-K recommendations with item features.</p>
{code('python recommend.py --user 42')}
{shots.section(['06_*.png'], 'Prediction demo', 'python recommend.py --user 42')}
</section>""")

    # ---------------------------------------------------------------- 11 conclusions
    S.append(f"""
<section><h2>11. Conclusions and limitations</h2>
<ul>
<li>All eight stages run end-to-end as an orchestrated, reproducible DAG; every stage writes inspectable outputs and summaries.</li>
<li>Validation caught every injected problem ({val['error_rows_total']} rejected records) and documented each failure with its rule.</li>
<li>The best model by NDCG@{k} is <b>{esc(PRETTY[best['model']])}</b> ({best[f'ndcg_at_{k}']:.4f}, {lift:+.1f}% vs the popularity baseline).</li>
<li><b>Limitations:</b> one global time split with offline metrics only (a rolling backtest or online A/B test would be more reliable);
{split['cold_start_test_users']} test users are cold-start and get popularity recommendations; MovieLens 100K is small and dense; MF optimises
rating error rather than ranking.</li>
<li><b>Future work:</b> ranking objectives (BPR / learning-to-rank), demographic cold-start models, Feast or a database-backed
feature store, and multi-seed / multi-window evaluation.</li>
</ul></section>""")

    # ---------------------------------------------------------------- appendix
    S.append(f"""
<section><h2>Appendix - full feature catalog ({len(catalog)} features)</h2>
{table(catalog[['name', 'entity', 'dtype', 'description', 'logic', 'used_by', 'null_pct']], 'small')}
</section>""")

    css = """
@page { size: A4; margin: 13mm 12mm; }
body { font-family: 'Segoe UI', Arial, sans-serif; color: #1c2430; font-size: 10pt; line-height: 1.45; max-width: 1000px; margin: 0 auto; }
h1 { font-size: 24pt; color: #17365d; margin: 10px 0 6px; line-height: 1.2; }
h2 { font-size: 15pt; color: #17365d; border-bottom: 2px solid #2f6db5; padding-bottom: 3px; margin-top: 0; }
h3 { font-size: 11pt; color: #24456e; margin: 14px 0 5px; page-break-after: avoid; }
section { page-break-before: always; padding-top: 2px; }
section.cover { page-break-before: avoid; }
.course { color: #2f6db5; font-weight: 600; letter-spacing: .3px; margin-top: 30px; }
.sub { color: #55657a; margin-bottom: 18px; }
.tbl { border-collapse: collapse; width: 100%; margin: 6px 0 10px; font-size: 8.4pt; page-break-inside: auto; }
.tbl th, .tbl td { border: 1px solid #c9d3df; padding: 3px 6px; text-align: left; vertical-align: top; }
.tbl th { background: #e9eff7; } .tbl tr { page-break-inside: avoid; }
.tbl.small { font-size: 7.4pt; }
.kv th { width: 34%; } .cover-kv { font-size: 10pt; } .cover-kv ul { margin: 0; padding-left: 18px; }
.kpis { display: grid; grid-template-columns: repeat(3, 1fr); gap: 8px; margin: 16px 0; }
.kpis div { border: 1px solid #c9d3df; border-radius: 6px; padding: 8px 10px; background: #f6f9fc; }
.kpis b { display: block; font-size: 16pt; color: #17365d; } .kpis span { color: #55657a; font-size: 8.5pt; }
.toc li { margin: 2px 0; }
figure { margin: 8px 0 12px; page-break-inside: avoid; text-align: center; }
figure img { max-width: 100%; max-height: 235mm; border: 1px solid #c9d3df; }
figcaption { font-size: 8.5pt; color: #55657a; margin-top: 3px; font-style: italic; }
pre.cmd { background: #1f2933; color: #e6edf3; padding: 8px 10px; border-radius: 5px; font-size: 8.5pt; white-space: pre-wrap; }
code { background: #eef2f7; padding: 0 3px; border-radius: 3px; font-size: 8.6pt; }
.placeholder { border: 2px dashed #d9822b; background: #fff7ee; padding: 10px; margin: 8px 0; font-size: 9pt; page-break-inside: avoid; }
a { color: #2f6db5; }
"""
    return (f"<!DOCTYPE html><html><head><meta charset='utf-8'><title>DM4ML Assignment 2 - Submission Report</title>"
            f"<style>{css}</style></head><body>{''.join(S)}</body></html>")


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description="Build reports/submission_report.html")
    ap.add_argument("--group", default="Individual submission", help='e.g. "Individual submission" or "Group 12"')
    ap.add_argument("--member", action="append", default=[], help='repeatable: "Full Name (ID)"')
    ap.add_argument("--github", default="https://github.com/<user>/recommendation-pipeline")
    args = ap.parse_args()
    cfg = load_config()
    out = cfg["paths"]["reports"] / "submission_report.html"
    out.write_text(build(cfg, args), encoding="utf-8")
    missing = [f"stage{i}" for i in range(1, 9) if not list(cfg["paths"]["figures"].glob(f"stage{i}_*.png"))]
    print(f"Written: {out}")
    if missing:
        print("Missing stage screenshots (placeholders shown): " + ", ".join(missing))
    print("Open it in Chrome -> Ctrl+P -> Destination 'Save as PDF' -> More settings: Background graphics ON -> Save.")


if __name__ == "__main__":
    main()
