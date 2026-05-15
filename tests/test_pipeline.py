import os
import pandas as pd
import joblib


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

    required_columns = [
        "y_1",
        "y_3",
        "y_5",
        "y_7",
    ]

    for col in required_columns:
        assert col in df.columns


def test_model_artifact_exists():
    """
    Ensure trained model artifact exists.
    """
    assert os.path.exists("models/best_model.pkl")


def test_scaler_artifact_exists():
    """
    Ensure scaler artifact exists.
    """
    assert os.path.exists("models/scaler.pkl")


def test_model_can_be_loaded():
    """
    Ensure saved model can be loaded correctly.
    """
    model = joblib.load("models/best_model.pkl")
    assert model is not None


def test_results_file_exists():
    """
    Ensure evaluation results are generated.
    """
    assert os.path.exists("models/model_results.csv")


def test_results_have_metrics():
    """
    Ensure evaluation metrics exist in results file.
    """
    results = pd.read_csv("models/model_results.csv")

    required_metrics = ["rmse", "mae", "r2"]

    for metric in required_metrics:
        assert metric in results.columns
