import os
import pandas as pd
import joblib
import pytest


def _ml_artifacts_exist() -> bool:
    return os.path.exists("models/best_model.pkl") and os.path.exists(
        "models/scaler.pkl"
    )


def _arima_results_exist() -> bool:
    return os.path.exists("models/arima_results.csv")


def test_processed_dataset_exists():
    """
    Ensure processed dataset is generated correctly.
    """
    assert os.path.exists("data/processed/gold_features.csv")


def test_processed_dataset_not_empty():
    """
    Ensure processed dataset contains rows.
    """
    df = pd.read_csv("data/processed/gold_features.csv")
    assert len(df) > 0


def test_required_columns_exist():
    """
    Ensure important columns exist after preprocessing.
    """
    df = pd.read_csv("data/processed/gold_features.csv")

    multi_horizon_cols = ["y_1", "y_3", "y_5", "y_7"]

    if all(col in df.columns for col in multi_horizon_cols):
        return

    assert "target" in df.columns, (
        "Expected multi-horizon targets (y_1, y_3, y_5, y_7) or a single 'target' column."
    )


def test_model_or_arima_artifact_exists():
    """
    Ensure at least one trained artifact exists (ML or ARIMA).
    """
    assert _ml_artifacts_exist() or _arima_results_exist()


def test_model_can_be_loaded():
    """
    Ensure saved model can be loaded correctly.
    """
    if not _ml_artifacts_exist():
        pytest.skip("ML artifacts not found; ARIMA results present instead.")

    model = joblib.load("models/best_model.pkl")
    assert model is not None


def test_results_file_exists():
    """
    Ensure evaluation results are generated.
    """
    assert os.path.exists("models/model_results.csv") or os.path.exists(
        "models/arima_results.csv"
    )


def test_results_have_metrics():
    """
    Ensure evaluation metrics exist in results file.
    """
    if os.path.exists("models/model_results.csv"):
        results = pd.read_csv("models/model_results.csv")
    elif os.path.exists("models/arima_results.csv"):
        results = pd.read_csv("models/arima_results.csv")
    else:
        pytest.skip("No results file found.")

    required_metrics = ["rmse", "mae", "r2"]

    for metric in required_metrics:
        assert metric in results.columns
