"""Contrôles de sélection préparés pour une exécution ultérieure, sans fit coûteux."""

import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import average_precision_score, log_loss


@pytest.fixture
def api():
    path = Path(__file__).resolve().parents[1] / "notebooks/telco_churn_modeling.ipynb"
    cells = json.loads(path.read_text())["cells"]
    namespace = {
        "Path": Path, "pd": pd, "np": np, "SEED": 42,
        "LABEL_DELAY": pd.Timedelta(30, unit="D"),
        "VALIDATION_MONTHS": ("2024-07", "2024-08", "2024-09"),
        "average_precision_score": average_precision_score, "log_loss": log_loss,
    }
    for cell in cells:
        if cell["cell_type"] == "code":
            definitions = [node for node in ast.parse("".join(cell["source"])).body if isinstance(node, ast.FunctionDef)]
            exec(compile(ast.Module(body=definitions, type_ignores=[]), "definitions", "exec"), namespace)
    return namespace


def test_quality_distinguishes_constants_from_missingness(api):
    data = pd.DataFrame({
        "churn": [0, 0, 1, 1], "category": ["a", "a", "b", "b"],
        "numeric": [0., 1., 2., 3.], "constant": 1.,
        "missing": np.nan, "pattern": [1., np.nan, 1., np.nan],
    })
    categories, details = api["analyze_categorical_features"](data, ["category"])
    assert categories.loc["category", "cramer_v"] == pytest.approx(1.)
    assert details["category"]["observations"].sum() == len(data)
    summary, groups = api["analyze_numeric_features"](data, ["numeric", "constant", "missing", "pattern"])
    assert summary.loc["constant", "constant"]
    assert summary.loc["missing", "all_missing"]
    assert not summary.loc["pattern", "constant"]
    assert summary.loc["numeric", "spearman_churn"] > .8
    for table in groups.values():
        assert table["observations"].sum() == len(data)


def test_numeric_groups_are_ordered_and_do_not_mutate_input(api):
    data = pd.DataFrame({"churn": [0, 1] * 10 + [0], "value": list(range(20)) + [np.nan]})
    original = data.copy(deep=True)
    _, groups = api["analyze_numeric_features"](data, ["value"])
    pd.testing.assert_frame_equal(data, original)
    table = groups["value"]
    assert table.iloc[-1]["groupe"] == "Non renseigné"
    assert len(table) == 6
    assert table["observations"].sum() == 21


def test_candidate_proposals_are_not_automatic_removals(api):
    quality = pd.DataFrame({"all_missing": [False, False, True], "constant": [False, False, True]}, index=["a", "b", "c"])
    permutations = pd.DataFrame({"feature": ["a", "a", "b", "b", "c", "c"], "ap_drop": [.1, .2, 0., 0., .1, .1]})
    candidates, summary = api["propose_candidates"](quality, permutations, .001, [])
    assert set(candidates) == {"b", "c"}
    assert not summary.loc["a", "candidat"]
    with pytest.raises(ValueError, match="inconnus"):
        api["propose_candidates"](quality, permutations, .001, ["absent"])


def test_sequential_ablation_rejects_a_useful_variable(api):
    def make_experiment(features, probabilities):
        monthly = {}
        for month in api["VALIDATION_MONTHS"]:
            predictions = pd.DataFrame({"churn": [0, 1], "probability": probabilities})
            monthly[month] = {"predictions": predictions}
        return {
            "name": "Référence", "data": pd.DataFrame(), "features": features.copy(),
            "categorical_features": [], "monthly_results": monthly, "threshold": .5,
            "validation_predictions": pd.concat([item["predictions"] for item in monthly.values()], ignore_index=True),
        }

    reference = make_experiment(["noise", "signal", "backup"], [.1, .9])
    tried = []
    def fake_validation(name, data, features, categories):
        tried.append(features.copy())
        probabilities = [.1, .9] if "signal" in features else [.6, .4]
        return make_experiment(features, probabilities)
    api["run_validation"] = fake_validation
    selected, history = api["select_by_ablation"](reference, ["noise", "signal"], .001, .001)
    assert selected["features"] == ["signal", "backup"]
    assert reference["features"] == ["noise", "signal", "backup"]
    assert history["retirée"].tolist() == [True, False]
    assert tried == [["signal", "backup"], ["backup"]]
    unchanged, _ = api["select_by_ablation"](reference, [], .001, .001)
    assert unchanged["monthly_results"] is not reference["monthly_results"]


def test_permutation_uses_only_validation_and_never_fits(api):
    class Model:
        def fit(self, *args, **kwargs):
            raise AssertionError("La permutation ne doit pas entraîner de modèle")
        def predict_proba(self, x):
            probability = x["signal"].to_numpy()
            return np.column_stack([1-probability, probability])
    rows = []
    for month in api["VALIDATION_MONTHS"]:
        for label in [0, 1, 0, 1]:
            rows.append({"snapshot_date": pd.Period(month, freq="M").end_time.normalize(),
                         "churn": label, "signal": .9 if label else .1, "noise": 1.})
    data = pd.DataFrame(rows)
    def fake_fold(data, scoring_date):
        return {"score": data.loc[data.snapshot_date == scoring_date]}
    api["build_monthly_fold"] = fake_fold
    experiment = {"data": data, "features": ["signal", "noise"], "categorical_features": [],
                  "monthly_results": {month: {"model": Model()} for month in api["VALIDATION_MONTHS"]}}
    result = api["validation_permutation_importance"](experiment, repeats=2)
    assert set(result["mois"]) == set(api["VALIDATION_MONTHS"])
    assert len(result) == 3 * 2 * 2
    assert result.loc[result.feature == "noise", "ap_drop"].eq(0).all()
    assert result.loc[result.feature == "signal", "ap_drop"].gt(0).any()
