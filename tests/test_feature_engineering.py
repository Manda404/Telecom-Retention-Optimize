"""Régressions métier : tester les définitions du notebook sans exécuter son workflow."""

import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def feature_engineer_class():
    notebook = json.loads((ROOT / "notebooks/01_churn_feature_engineering.ipynb").read_text())
    source = next(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code" and "class ChurnFeatureEngineer:" in "".join(cell["source"])
    )
    namespace = {"pd": pd, "np": np}
    exec(compile(source, "notebook-feature-definitions", "exec"), namespace)
    return namespace["ChurnFeatureEngineer"]


@pytest.fixture
def panel():
    return pd.DataFrame(
        {
            "customer_id": ["a"] * 9,
            "snapshot_date": pd.to_datetime(
                [
                    "2024-01-31",
                    "2024-02-29",
                    "2024-03-31",
                    "2024-04-30",
                    "2024-05-31",
                    "2024-06-30",
                    "2024-07-31",
                    "2024-08-31",
                    "2024-10-31",
                ]
            ),
            "signup_date": pd.Timestamp("2020-01-01"),
            "monthly_fee": [40.0, 48.0, 48.0, 48.0, 0.0, 12.0, np.nan, 20.0, 30.0],
            "data_gb": [10.0, 20.0, 30.0, 10.0, 0.0, 0.0, np.nan, 0.0, 0.0],
            "support_calls_3m": [0.0, 1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "nps": [8.0, np.nan, np.nan, 5.0, np.nan, np.nan, np.nan, np.nan, np.nan],
            "churn": [0] * 9,
        },
        index=[91, 12, 81, 16, 21, 11, 100, 43, 8],
    )


def test_business_windows_and_gaps(feature_engineer_class, panel):
    result = feature_engineer_class().fit(panel.iloc[:3]).transform(panel)
    np.testing.assert_allclose(
        result.monthly_fee_change_pct,
        [np.nan, 20, 0, 0, -100, np.nan, np.nan, np.nan, np.nan],
        equal_nan=True,
    )
    np.testing.assert_allclose(result.usage_change_vs_recent_pct.iloc[[3, 4]], [-50, -100])
    np.testing.assert_allclose(
        result.zero_usage_streak, [0, 0, 0, 0, 1, 2, np.nan, 1, 1], equal_nan=True
    )
    np.testing.assert_allclose(result.support_presence_3_snapshots.iloc[[2, 3, 4, 5]], [2, 3, 2, 1])


def test_satisfaction_freshness_is_not_filled_from_future(feature_engineer_class, panel):
    panel.loc[panel.index[:2], "nps"] = np.nan
    panel.loc[panel.index[2], "nps"] = 7
    result = feature_engineer_class().fit(panel.iloc[:3]).transform(panel)
    assert result.last_observed_nps.iloc[:2].isna().all()
    assert result.months_since_observed_nps.iloc[:2].isna().all()
    assert result.last_observed_nps.iloc[2] == 7
    assert result.months_since_observed_nps.iloc[2] == 0
    assert result.months_since_observed_nps.iloc[-1] == 6


def test_clients_and_future_cannot_contaminate_past(feature_engineer_class, panel):
    other = panel.copy()
    other.customer_id = "b"
    other.monthly_fee = 60.0
    other.nps = np.nan
    other.data_gb = 0.0
    context = pd.concat([panel, other], ignore_index=True).sample(frac=1, random_state=1)
    engineer = feature_engineer_class().fit(panel.iloc[:3])
    baseline = engineer.transform(context)
    changed = context.copy()
    future = changed.snapshot_date.ge("2024-07-01")
    changed.loc[future, ["monthly_fee", "data_gb", "support_calls_3m", "nps"]] = 999
    after = engineer.transform(changed)
    pd.testing.assert_frame_equal(
        baseline.loc[~future, engineer.created_features],
        after.loc[~future, engineer.created_features],
    )
    assert baseline.loc[context.customer_id.eq("b"), "last_observed_nps"].isna().all()
    assert "churn" not in engineer._history


def test_transform_preserves_input_and_frozen_bounds(feature_engineer_class, panel):
    before = panel.copy(deep=True)
    engineer = feature_engineer_class().fit(panel.iloc[:3])
    metadata = engineer.get_feature_metadata()
    result = engineer.transform(panel.sample(frac=1, random_state=2))
    assert result.index.tolist() == panel.sample(frac=1, random_state=2).index.tolist()
    pd.testing.assert_frame_equal(panel, before)
    assert engineer.get_feature_metadata() == metadata
    with pytest.raises(ValueError, match="fit"):
        feature_engineer_class().transform(panel)


def test_normalization_and_ambiguous_keys(tmp_path):
    notebook = json.loads((ROOT / "notebooks/01_churn_feature_engineering.ipynb").read_text())
    source = next(
        "".join(c["source"])
        for c in notebook["cells"]
        if c["cell_type"] == "code" and "def read_telco_source(" in "".join(c["source"])
    )
    function = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef))
    namespace = {"pd": pd, "np": np}
    exec(compile(ast.Module(body=[function], type_ignores=[]), "reader", "exec"), namespace)
    base = dict(
        customer_id="001",
        snapshot_date="2024-01-31",
        signup_date="2020-01-01",
        contract=" MENSUEL ",
        region="ARA",
        dept="01",
        channel="web",
        monthly_fee=-10,
        data_gb=2,
        support_calls_3m=0,
        late_payments=0,
        nps=99,
        churn=0,
    )
    rows = [
        base,
        base.copy(),
        {**base, "customer_id": "002", "churn": 0},
        {**base, "customer_id": "002", "churn": 1},
    ]
    path = tmp_path / "source.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    original = path.read_bytes()
    _, clean, _, quality = namespace["read_telco_source"](path)
    assert clean.customer_id.tolist() == ["001"]
    assert clean.dept.iloc[0] == "01"
    assert clean.contract.iloc[0] == "mensuel"
    assert clean.monthly_fee.isna().all() and clean.nps.isna().all()
    assert quality["normalized_copies_removed"] == 1
    assert quality["ambiguous_keys_excluded"] == 1
    assert path.read_bytes() == original
