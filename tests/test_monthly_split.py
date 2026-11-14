"""Calendrier des labels et séparation train/early stopping/scoring, sans fit coûteux."""

import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def api():
    cells = json.loads(
        (Path(__file__).resolve().parents[1] / "notebooks/telco_churn_modeling.ipynb").read_text()
    )["cells"]
    namespace = {
        "Path": Path,
        "pd": pd,
        "np": np,
        "LABEL_DELAY": pd.Timedelta(30, unit="D"),
        "SEED": 42,
    }
    for cell in cells:
        if cell["cell_type"] != "code":
            continue
        definitions = [
            node
            for node in ast.parse("".join(cell["source"])).body
            if isinstance(node, ast.FunctionDef)
        ]
        exec(
            compile(ast.Module(body=definitions, type_ignores=[]), "notebook_definitions", "exec"),
            namespace,
        )
    return namespace


@pytest.fixture
def panel():
    return pd.DataFrame(
        [
            {
                "customer_id": str(client),
                "snapshot_date": date,
                "churn": int(client == 3),
                "contract": "mensuel",
                "monthly_fee": 30.0,
            }
            for date in pd.date_range("2024-01-31", "2024-11-30", freq="ME")
            for client in range(4)
        ]
    )


@pytest.mark.parametrize(
    "month,train_end,early_month",
    [
        ("2024-07", "2024-05", "2024-06"),
        ("2024-08", "2024-06", "2024-07"),
        ("2024-09", "2024-07", "2024-08"),
        ("2024-10", "2024-08", "2024-09"),
        ("2024-11", "2024-09", "2024-10"),
    ],
)
def test_calendar_and_disjoint_rows(api, panel, month, train_end, early_month):
    scoring_date = pd.Period(month, freq="M").end_time.normalize()
    fold = api["build_monthly_fold"](panel, scoring_date)
    assert fold["train"].snapshot_date.max().to_period("M") == pd.Period(train_end, freq="M")
    assert fold["early_stop"].snapshot_date.dt.to_period("M").eq(early_month).all()
    assert fold["score"].snapshot_date.dt.to_period("M").eq(month).all()
    for name in ["train", "early_stop"]:
        assert (fold[name].snapshot_date + pd.Timedelta(30, unit="D") <= scoring_date).all()
    assert not set(fold["train"].index) & set(fold["early_stop"].index)
    assert not set(fold["score"].index) & set(fold["train"].index)
    assert not set(fold["score"].index) & set(fold["early_stop"].index)


def test_exact_thirty_day_boundary(api, panel):
    # Septembre est mature le 30 octobre, pas le 29.
    with pytest.raises(ValueError, match="pas tous disponibles"):
        api["build_monthly_fold"](panel, pd.Timestamp("2024-10-29"))
    november = api["build_monthly_fold"](panel, pd.Timestamp("2024-11-30"))
    assert (
        (november["early_stop"].snapshot_date + pd.Timedelta(30, unit="D")).eq("2024-11-30").all()
    )
    # Janvier n'est pas encore mature le 29 février (année bissextile).
    with pytest.raises(ValueError, match="pas tous disponibles"):
        api["build_monthly_fold"](panel, pd.Timestamp("2024-02-29"))


def test_future_records_do_not_change_training(api, panel):
    date = pd.Timestamp("2024-09-30")
    full = api["build_monthly_fold"](panel, date)
    truncated = api["build_monthly_fold"](panel.loc[panel.snapshot_date <= date], date)
    for name in full:
        pd.testing.assert_frame_equal(full[name], truncated[name])
    modified = panel.copy()
    modified.loc[modified.snapshot_date > date, "churn"] = 1
    same = api["build_monthly_fold"](modified, date)
    for name in ["train", "early_stop"]:
        pd.testing.assert_frame_equal(full[name], same[name])


def test_empty_month_or_single_training_class_fails(api, panel):
    without_june = panel.loc[panel.snapshot_date.dt.month.ne(6)]
    with pytest.raises(ValueError):
        api["build_monthly_fold"](without_june, pd.Timestamp("2024-07-31"))
    with pytest.raises(ValueError):
        api["build_monthly_fold"](panel.assign(churn=0), pd.Timestamp("2024-07-31"))


def test_fitted_and_early_stopping_indices_exclude_score(api, panel):
    class ModelSpy:
        def __init__(self, **parameters):
            self.parameters = parameters

        def fit(self, x, y, **kwargs):
            self.fit_indices = x.index
            self.fit_targets = y
            self.early_indices = kwargs["eval_set"][0].index
            self.early_targets = kwargs["eval_set"][1]

        def predict_proba(self, x):
            return np.tile([0.7, 0.3], (len(x), 1))

    fold = api["build_monthly_fold"](panel, pd.Timestamp("2024-11-30"))
    api["CatBoostClassifier"] = ModelSpy
    result = api["fit_predict_month"](
        panel, "2024-11", ["contract", "monthly_fee"], ["contract"],
    )
    model = result["model"]
    assert set(model.fit_indices) == set(fold["train"].index)
    assert set(model.early_indices) == set(fold["early_stop"].index)
    assert not set(result["predictions"].index) & set(model.fit_indices)
    assert not set(result["predictions"].index) & set(model.early_indices)
    weights = result["class_weights"]
    assert weights[1] / weights[0] == pytest.approx(3.0)


@pytest.mark.parametrize(
    "labels,probabilities",
    [
        ([0, 1, 0, 1], [0.1, 0.8, 0.8, 0.9]),
        ([0, 1, 0, 1], [0.2, 0.2, 0.2, 0.2]),
    ],
)
def test_simple_threshold_loop_matches_f1_reference(api, labels, probabilities):
    from sklearn.metrics import f1_score

    predictions = pd.DataFrame({"churn": labels, "probability": probabilities})
    from sklearn.metrics import precision_recall_curve
    api["precision_recall_curve"] = precision_recall_curve
    threshold = api["choose_threshold"](predictions)
    candidates = []
    for candidate in np.unique(probabilities):
        predicted = np.asarray(probabilities) >= candidate
        candidates.append((f1_score(labels, predicted, zero_division=0), candidate))
    assert threshold == max(candidates)[1]
