"""Declarative Pandera schemas (second, independent validation layer).

The rule engine in ``checks.py`` decides which rows go to the error dataset.
These Pandera schemas re-validate the *raw* tables declaratively and their
failure summary is added to the validation report, which documents the data
contract in a standard, tool-based way. If Pandera is not installed the
report simply notes that this layer was skipped.
"""
from __future__ import annotations

from typing import Dict, List

import pandas as pd

try:  # pandera >= 0.20 exposes the pandas API under pandera.pandas
    import pandera.pandas as pa  # type: ignore
except Exception:  # pragma: no cover
    try:
        import pandera as pa  # type: ignore
    except Exception:
        pa = None

PANDERA_OK = pa is not None


def build_schemas(vcfg: Dict, genre_cols: List[str]):
    """Return Pandera DataFrameSchemas for users, items and interactions (coercing text to types)."""
    users = pa.DataFrameSchema({
        "user_id": pa.Column(int, coerce=True, nullable=False, unique=True),
        "age": pa.Column(int, pa.Check.in_range(vcfg["min_age"], vcfg["max_age"]), coerce=True, nullable=False),
        "gender": pa.Column(str, pa.Check.isin(vcfg["allowed_genders"]), nullable=False),
        "occupation": pa.Column(str, nullable=True),
        "zip_code": pa.Column(str, nullable=True),
    })
    items = pa.DataFrameSchema({
        "item_id": pa.Column(int, coerce=True, nullable=False, unique=True),
        "title": pa.Column(str, nullable=False),
        **{g: pa.Column(int, pa.Check.isin([0, 1]), coerce=True) for g in genre_cols},
    })
    interactions = pa.DataFrameSchema({
        "user_id": pa.Column(int, coerce=True, nullable=False),
        "item_id": pa.Column(int, coerce=True, nullable=False),
        "rating": pa.Column(float, pa.Check.isin([float(r) for r in range(vcfg["rating_min"], vcfg["rating_max"] + 1)]),
                            coerce=True, nullable=False),
        "timestamp": pa.Column(int, pa.Check.ge(int(pd.Timestamp(vcfg["min_timestamp"]).value // 10**9)),
                               coerce=True, nullable=False),
    })
    return {"users": users, "items": items, "interactions": interactions}


def run_pandera(tables: Dict[str, pd.DataFrame], vcfg: Dict, genre_cols: List[str]) -> pd.DataFrame:
    """Validate raw tables lazily and return a summary of failures by (table, column, check)."""
    if not PANDERA_OK:
        return pd.DataFrame([{"table": "-", "column": "-", "check": "pandera not installed - layer skipped", "failures": 0}])
    rows = []
    for name, schema in build_schemas(vcfg, genre_cols).items():
        df = tables[name][list(schema.columns)].copy()
        try:
            schema.validate(df, lazy=True)
            rows.append({"table": name, "column": "*", "check": "all schema checks", "failures": 0})
        except pa.errors.SchemaErrors as err:
            fc = err.failure_cases
            fc = fc.assign(column=fc["column"].astype(str), check=fc["check"].astype(str))
            agg = fc.groupby(["column", "check"]).size().reset_index(name="failures")
            rows += [{"table": name, **r} for r in agg.to_dict("records")]
        except Exception as exc:  # never break the pipeline because of the second layer
            rows.append({"table": name, "column": "*", "check": f"pandera error: {exc}", "failures": -1})
    return pd.DataFrame(rows)
