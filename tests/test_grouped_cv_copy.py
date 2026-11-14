"""Contrôler les groupes, les colonnes et les métriques sans entraînement coûteux."""

import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (
    average_precision_score, roc_auc_score, log_loss,
    f1_score, precision_score, recall_score, matthews_corrcoef,
)


@pytest.fixture
def api():
    path = Path(__file__).resolve().parents[1] / "notebooks/telco_churn_feature_selection_grouped_cv.ipynb"
    cells = json.loads(path.read_text())["cells"]
    namespace = {
        "Path": Path, "pd": pd, "np": np, "SEED": 42,
        "HOLDOUT_SPLITS": 5, "CV_SPLITS": 4,
        "MAX_ITERATIONS": 500,
        "MODEL_FEATURES": [
            "contract", "channel", "monthly_fee", "data_gb", "support_calls_3m",
            "late_payments", "nps", "last_invoice_amount",
        ],
        "CATEGORICAL_FEATURES": ["contract", "channel"],
        "NUMERIC_FEATURES": [
            "monthly_fee", "data_gb", "support_calls_3m", "late_payments", "nps",
            "last_invoice_amount",
        ],
        "StratifiedGroupKFold": StratifiedGroupKFold,
        "average_precision_score": average_precision_score, "roc_auc_score": roc_auc_score,
        "log_loss": log_loss, "f1_score": f1_score, "precision_score": precision_score,
        "recall_score": recall_score, "matthews_corrcoef": matthews_corrcoef,
        "THRESHOLD": 0.5,
    }
    for cell in cells:
        if cell["cell_type"] == "code":
            definitions = [node for node in ast.parse("".join(cell["source"])).body if isinstance(node, ast.FunctionDef)]
            exec(compile(ast.Module(body=definitions, type_ignores=[]), "grouped_definitions", "exec"), namespace)
    return namespace


@pytest.fixture
def panel():
    rows = []
    dates = pd.date_range("2024-01-31", periods=3, freq="ME")
    for client in range(100):
        for month, date in enumerate(dates):
            rows.append({
                "customer_id": f"C{client}", "snapshot_date": date,
                "churn": int(month == 2), "contract": "mensuel", "monthly_fee": 50.,
            })
    return pd.DataFrame(rows)


def test_group_and_row_integrity(api, panel):
    splits = api["make_grouped_splits"](panel)
    development_ids = set(splits["development"].customer_id)
    test_ids = set(splits["test"].customer_id)
    assert development_ids.isdisjoint(test_ids)
    assert development_ids | test_ids == set(panel.customer_id)
    evaluated_ids = set()
    evaluated_rows = set()
    for parts in splits["cv_splits"].values():
        api["assert_group_separation"](parts)
        for data in parts.values():
            assert set(data.customer_id).isdisjoint(test_ids)
            assert data.groupby("customer_id").size().eq(3).all()
        score_ids = set(parts["score"].customer_id)
        assert evaluated_ids.isdisjoint(score_ids)
        evaluated_ids.update(score_ids)
        evaluated_rows.update(parts["score"].index)
    assert evaluated_ids == development_ids
    assert evaluated_rows == set(splits["development"].index)
    api["assert_group_separation"](splits["final_split"])
    assert set(splits["final_split"]["score"].customer_id) == test_ids


def test_reproducible_and_independent_of_pandas_index(api, panel):
    original = panel.copy(deep=True)
    first = api["make_grouped_splits"](panel)
    duplicate_index = panel.copy()
    duplicate_index.index = [7] * len(panel)
    second = api["make_grouped_splits"](duplicate_index)
    assert set(first["test"].customer_id) == set(second["test"].customer_id)
    for label in first["cv_splits"]:
        for role in ["train", "score"]:
            assert set(first["cv_splits"][label][role].customer_id) == set(second["cv_splits"][label][role].customer_id)
    pd.testing.assert_frame_equal(panel, original)


def test_invalid_partition_rejected(api, panel):
    with pytest.raises(ValueError):
        api["make_grouped_splits"](panel.assign(churn=0))
    with pytest.raises(ValueError):
        api["make_grouped_splits"](panel.assign(customer_id=None))
    same = panel.iloc[:6]
    with pytest.raises(AssertionError, match="Client partagé"):
        api["assert_group_separation"]({"train": same, "score": same.copy()})


def test_fit_never_receives_evaluated_clients(api, panel):
    class ModelSpy:
        def __init__(self, **parameters):
            self.parameters = parameters
        def fit(self, x, y, **kwargs):
            self.feature_names_ = x.columns.tolist()
            self.train_indices = x.index
            self.validation = kwargs["eval_set"]
            self.fit_options = kwargs
        def predict_proba(self, x):
            return np.tile([.7, .3], (len(x), 1))
    api["CatBoostClassifier"] = ModelSpy
    splits = api["make_grouped_splits"](panel)
    experiment = api["run_grouped_cv"](
        "Contrôle", splits["development"], splits["cv_splits"],
        ["contract", "monthly_fee"], ["contract"],
    )
    for label, result in experiment["cv_results"].items():
        parts = splits["cv_splits"][label]
        assert result["model"].parameters["iterations"] == 500
        assert result["features_used"] == ["contract", "monthly_fee"]
        assert set(result["model"].train_indices) == set(parts["train"].index)
        assert set(result["model"].validation[0].index) == set(parts["score"].index)
        assert result["model"].fit_options["use_best_model"] is False
        assert "early_stopping_rounds" not in result["model"].fit_options
        assert not set(parts["score"].index) & set(result["model"].train_indices)
        counts = parts["train"].churn.value_counts()
        assert result["class_weights"][0] * counts[0] == pytest.approx(result["class_weights"][1] * counts[1])
    before = experiment["threshold"]
    api["run_final_test"](experiment, splits["final_split"])
    assert experiment["threshold"] == before
    final_model = experiment["final_result"]["model"]
    assert final_model.validation is None, "Le test ne doit pas être transmis à fit."
    assert set(final_model.train_indices) == set(splits["development"].index)
    assert set(experiment["test_predictions"].customer_id) == set(splits["test"].customer_id)
    assert set(experiment["validation_predictions"].customer_id).isdisjoint(experiment["test_predictions"].customer_id)


def test_mcc_is_reported(api):
    perfect = api["evaluate"](np.array([0, 0, 1, 1]), np.array([.1, .2, .8, .9]), .5)
    inverse = api["evaluate"](np.array([0, 0, 1, 1]), np.array([.9, .8, .2, .1]), .5)
    assert perfect["MCC"] == pytest.approx(1.)
    assert inverse["MCC"] == pytest.approx(-1.)


def test_manual_selection_is_applied_without_model(api):
    def forbidden_training(*args, **kwargs):
        raise AssertionError("Le choix de colonnes ne doit pas entraîner un modèle")
    api["CatBoostClassifier"] = forbidden_training
    api["run_grouped_cv"] = forbidden_training
    available = ["contract", "channel", "monthly_fee", "nps"]
    result = api["select_features"](["monthly_fee", "contract"], available, ["contract", "channel"])
    assert result["features"] == ["monthly_fee", "contract"]
    assert result["categorical_features"] == ["contract"]
    decisions = result["decisions"].set_index("feature")["décision"]
    assert decisions["channel"] == "Retirée"
    assert decisions["nps"] == "Retirée"
    assert decisions["monthly_fee"] == "Conservée"


def test_empty_or_invalid_manual_selection_is_explicit(api):
    available = ["contract", "monthly_fee"]
    for choice in [[], ["contract", "contract"], ["customer_id"]]:
        with pytest.raises(ValueError):
            api["select_features"](choice, available, ["contract"])


def test_fixed_threshold_replaces_optimization(api, panel):
    class ModelSpy:
        def __init__(self, **parameters):
            self.parameters = parameters
        def fit(self, x, y, **kwargs):
            self.feature_names_ = x.columns.tolist()
        def predict_proba(self, x):
            return np.tile([.7, .3], (len(x), 1))
    api["CatBoostClassifier"] = ModelSpy
    splits = api["make_grouped_splits"](panel)
    result = api["run_grouped_cv"](
        "Contrôle", splits["development"], splits["cv_splits"],
        ["monthly_fee"], [],
    )
    assert result["threshold"] == .5
    assert "choose_threshold" not in api
    result["threshold"] = .7
    with pytest.raises(ValueError, match="seuil différent"):
        api["run_final_test"](result, splits["final_split"])


def test_default_runs_reference_without_inventing_selection():
    path = Path(__file__).resolve().parents[1] / "notebooks/telco_churn_feature_selection_grouped_cv.ipynb"
    cells = json.loads(path.read_text())["cells"]
    source = next("".join(cell["source"]) for cell in cells
                  if cell["cell_type"] == "code" and "SELECTED_FEATURES =" in "".join(cell["source"]))
    namespace = {"MODEL_FEATURES": ["contract", "monthly_fee"], "display": lambda *args: None, "pd": pd}
    exec(source, namespace)
    assert namespace["feature_sets"] == {"Référence": ["contract", "monthly_fee"]}


def test_cleaning_reports_missing_and_conflicting_months(api, tmp_path):
    rows = []
    for client in ["A", "B", "C", "D"]:
        row = {feature: 1 for feature in api["MODEL_FEATURES"]}
        row.update(customer_id=client, snapshot_date="2024-01-31", churn=0,
                   contract=" 'Mensuel' ", channel="WEB", nps=11, monthly_fee=-2)
        rows.append(row)
    rows[3]["churn"] = np.nan
    rows.extend([rows[0].copy(), dict(rows[1], monthly_fee=50)])
    path = tmp_path / "panel.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    cleaned = api["prepare_data"](path)
    assert set(cleaned.customer_id) == {"A", "C"}
    assert cleaned["nps"].isna().all()
    assert cleaned["monthly_fee"].isna().all()
    assert cleaned["contract"].eq("mensuel").all()
    assert cleaned["channel"].eq("web").all()
