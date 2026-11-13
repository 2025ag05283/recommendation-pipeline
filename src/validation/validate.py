"""Stage 2 - Data validation.

Reads the raw text tables, applies the rule engine (``checks.py``) in the
correct order (users and items first, so interactions can be checked for
referential integrity against the *valid* ids), then:

* writes valid, correctly typed tables to ``data/processed/*_validated``;
* writes rejected rows with their failed rules to ``data/errors/*_errors.csv``;
* runs the Pandera schemas as a second layer;
* writes ``reports/validation_report.md`` + ``reports/validation_summary.json``;
* enforces a quality gate (``validation.max_error_rate``).
"""
from __future__ import annotations

import time
from typing import Dict

import pandas as pd

from src.ingestion.ingest import GENRES
from src.utils.io import save_json, save_table
from src.utils.logger import get_logger
from src.validation.checks import apply_checks, interaction_checks, is_blank, item_checks, user_checks
from src.validation.schemas import PANDERA_OK, run_pandera

log = get_logger(__name__)
GENRE_COLS = [f"genre_{g}" for g in GENRES]


class DataQualityError(RuntimeError):
    """Raised when the share of invalid interactions exceeds the configured threshold."""


def _type_users(df: pd.DataFrame) -> pd.DataFrame:
    """Cast validated user rows to their proper dtypes."""
    out = df.copy()
    out["user_id"] = pd.to_numeric(out["user_id"]).astype("int64")
    out["age"] = pd.to_numeric(out["age"]).astype("int64")
    for c in ["gender", "occupation", "zip_code"]:
        out[c] = out[c].astype("string")
    return out


def _type_items(df: pd.DataFrame) -> pd.DataFrame:
    """Cast validated item rows to their proper dtypes."""
    out = df.copy()
    out["item_id"] = pd.to_numeric(out["item_id"]).astype("int64")
    out[GENRE_COLS] = out[GENRE_COLS].apply(pd.to_numeric).astype("int64")
    for c in ["title", "release_date", "video_release_date", "imdb_url"]:
        out[c] = out[c].astype("string")
    return out


def _type_interactions(df: pd.DataFrame) -> pd.DataFrame:
    """Cast validated interaction rows to their proper dtypes."""
    out = df.copy()
    for c in ["user_id", "item_id", "rating", "timestamp"]:
        out[c] = pd.to_numeric(out[c]).astype("int64")
    return out


def _write_report(cfg: Dict, results: pd.DataFrame, counts: Dict, warnings: pd.DataFrame,
                  pandera_df: pd.DataFrame, path) -> None:
    """Render the human-readable validation report (Markdown)."""
    lines = ["# Data Validation Report", "",
             f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}  ",
             f"Quality gate: max invalid-interaction rate = {cfg['validation']['max_error_rate']:.0%}", "",
             "## 1. Row counts", "", "| Table | Raw rows | Valid rows | Rejected rows | Rejected % |",
             "|---|---:|---:|---:|---:|"]
    for t, c in counts.items():
        lines.append(f"| {t} | {c['raw']:,} | {c['valid']:,} | {c['rejected']:,} | {c['rejected_pct']:.3f}% |")
    lines += ["", "## 2. Rule-based checks (rows failing each rule)", "",
              "| Table | Check | Category | Rule | Failed rows | Status |", "|---|---|---|---|---:|---|"]
    for r in results.to_dict("records"):
        lines.append(f"| {r['table']} | `{r['check']}` | {r['category']} | {r['rule']} | {r['failed_rows']} | {r['status']} |")
    lines += ["", "## 3. Warnings - missing optional values (kept, imputed in preparation)", "",
              "| Table | Column | Missing values |", "|---|---|---:|"]
    for r in warnings.to_dict("records"):
        lines.append(f"| {r['table']} | {r['column']} | {r['missing']} |")
    lines += ["", "## 4. Pandera schema validation (independent declarative layer)", "",
              f"Pandera available: **{PANDERA_OK}**", "", "| Table | Column | Check | Failures |", "|---|---|---|---:|"]
    for r in pandera_df.to_dict("records"):
        lines.append(f"| {r['table']} | {r['column']} | {str(r['check']).replace('|', '/')} | {r['failures']} |")
    lines += ["", "## 5. Outputs", "",
              "* Valid typed tables: `data/processed/users_validated`, `items_validated`, `interactions_validated`",
              "* Error datasets (original values + `error_reasons`): `data/errors/users_errors.csv`, "
              "`items_errors.csv`, `interactions_errors.csv`, combined in `all_errors.csv`", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def run_validation(cfg: Dict) -> Dict:
    """Run the validation stage and return a summary dict."""
    raw, proc, err_dir, rep = (cfg["paths"][k] for k in ("raw", "processed", "errors", "reports"))
    fmt, vcfg = cfg["storage"]["format"], cfg["validation"]
    users = pd.read_csv(raw / "users.csv", dtype=str)
    items = pd.read_csv(raw / "items.csv", dtype=str)
    inter = pd.read_csv(raw / "interactions.csv", dtype=str)

    # 1) users and items first ...
    u_ok, u_err, u_res = apply_checks(users, user_checks(vcfg), "users")
    i_ok, i_err, i_res = apply_checks(items, item_checks(GENRE_COLS), "items")
    u_ok, i_ok = _type_users(u_ok), _type_items(i_ok)
    # 2) ... then interactions, with referential integrity against the VALID ids
    x_ok, x_err, x_res = apply_checks(inter, interaction_checks(vcfg, set(u_ok.user_id), set(i_ok.item_id)), "interactions")
    x_ok = _type_interactions(x_ok)

    # save valid tables and error datasets
    save_table(u_ok, proc / "users_validated", fmt)
    save_table(i_ok, proc / "items_validated", fmt)
    save_table(x_ok, proc / "interactions_validated", fmt)
    for name, e in [("users", u_err), ("items", i_err), ("interactions", x_err)]:
        e.to_csv(err_dir / f"{name}_errors.csv", index=False)
    def _flatten(e: pd.DataFrame) -> pd.DataFrame:
        """One line per rejected record: table, failed rules, and the original values joined by ' | '."""
        vals = e.drop(columns=["table", "error_reasons"])
        record = [" | ".join("" if pd.isna(v) else str(v) for v in row) for row in vals.itertuples(index=False)]
        return pd.DataFrame({"table": e["table"].tolist(), "error_reasons": e["error_reasons"].tolist(), "record": record})
    all_err = pd.concat([_flatten(e) for e in (u_err, i_err, x_err)], ignore_index=True)
    all_err.to_csv(err_dir / "all_errors.csv", index=False)

    # warnings: missing OPTIONAL fields in valid rows (handled in preparation)
    warn = [{"table": "users", "column": c, "missing": int(is_blank(u_ok[c]).sum())} for c in ["occupation", "zip_code"]]
    warn += [{"table": "items", "column": c, "missing": int(is_blank(i_ok[c]).sum())} for c in ["release_date"]]
    warnings = pd.DataFrame(warn)

    pandera_df = run_pandera({"users": users, "items": items, "interactions": inter}, vcfg, GENRE_COLS)
    results = pd.DataFrame(u_res + i_res + x_res)
    counts = {}
    for t, raw_df, ok in [("users", users, u_ok), ("items", items, i_ok), ("interactions", inter, x_ok)]:
        rej = len(raw_df) - len(ok)
        counts[t] = {"raw": len(raw_df), "valid": len(ok), "rejected": rej, "rejected_pct": 100 * rej / max(len(raw_df), 1)}
    _write_report(cfg, results, counts, warnings, pandera_df, rep / "validation_report.md")
    results.to_csv(rep / "validation_checks.csv", index=False)

    summary = {"counts": counts, "checks_failed": int((results.status == "FAIL").sum()),
               "checks_total": int(len(results)), "pandera_available": PANDERA_OK,
               "error_rows_total": int(len(all_err))}
    save_json(summary, rep / "validation_summary.json")
    log.info("Validation: %s", {t: f"{c['valid']}/{c['raw']} valid" for t, c in counts.items()})
    log.info("Rejected %d records -> data/errors/ ; report -> reports/validation_report.md", len(all_err))

    # quality gate
    rate = counts["interactions"]["rejected"] / max(counts["interactions"]["raw"], 1)
    if rate > vcfg["max_error_rate"]:
        raise DataQualityError(f"{rate:.2%} of interactions invalid (> {vcfg['max_error_rate']:.0%}) - stopping pipeline")
    return summary
