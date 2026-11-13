"""Unit tests for the validation rules and the time-based split."""
import pandas as pd

from src.preparation.prepare import time_based_split
from src.validation.checks import apply_checks, interaction_checks, user_checks

VCFG = {"rating_min": 1, "rating_max": 5, "min_timestamp": "1995-01-01", "min_age": 1, "max_age": 100,
        "allowed_genders": ["M", "F"]}


def test_invalid_interactions_are_separated_with_reasons():
    df = pd.DataFrame({
        "user_id": ["1", "1", None, "1", "99", "1", "1"],
        "item_id": ["10", "10", "10", "10", "10", "10", "10"],
        "rating": ["4", "4", "3", "7", "5", "five", "3"],
        "timestamp": ["880000000", "880000000", "880000001", "880000002", "880000003", "880000004", "4102444800"],
    })
    ok, err, results = apply_checks(df, interaction_checks(VCFG, {1}, {10}), "interactions")
    assert len(ok) == 1                                             # only the first row is valid
    reasons = ";".join(err["error_reasons"])
    for rule in ["no_exact_duplicates", "interaction_required_fields", "rating_valid", "user_exists",
                 "rating_numeric", "timestamp_plausible"]:
        assert rule in reasons


def test_user_rules():
    df = pd.DataFrame({"user_id": ["1", "1", "2", "3"], "age": ["30", "30", "-5", "40"],
                       "gender": ["M", "M", "F", "X"], "occupation": ["a", "a", "b", "c"], "zip_code": ["1"] * 4})
    ok, err, _ = apply_checks(df, user_checks(VCFG), "users")
    assert ok["user_id"].tolist() == ["1"]
    assert set(err["error_reasons"]) == {"user_id_unique", "age_in_range", "gender_allowed"}


def test_time_split_has_no_leakage():
    df = pd.DataFrame({"user_id": [1] * 10, "item_id": range(10), "rating": [3] * 10, "timestamp": range(100, 110)})
    train, test, _ = time_based_split(df, 0.2)
    assert train["timestamp"].max() < test["timestamp"].min()
    assert len(train) + len(test) == len(df)
