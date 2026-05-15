import os
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=FutureWarning)
import mlflow

mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow_new.db")
mlflow_artifact_root = os.getenv("MLFLOW_ARTIFACT_ROOT", str(Path.cwd() / "mlruns"))
Path(mlflow_artifact_root).mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MLFLOW_ARTIFACT_ROOT", str(Path(mlflow_artifact_root).resolve()))
mlflow.set_tracking_uri(mlflow_tracking_uri)
from mlflow.tracking import MlflowClient

EXPERIMENT_NAME = "gold-close-forecast"
HORIZONS = [1, 3, 5, 7]
MODEL_NAME_PREFIX = "gold-close-model"


def _best_run_id(horizon: int) -> str | None:
    runs = mlflow.search_runs(
        experiment_names=[EXPERIMENT_NAME],
        filter_string=f"tags.mlflow.runName LIKE 'BEST_%_h{horizon}_final'",
    )
    if runs.empty:
        return None

    metric_col = None
    for col in ("metrics.avg_rmse", "metrics.rmse"):
        if col in runs.columns:
            metric_col = col
            break

    if metric_col is None:
        return runs.iloc[0].run_id

    runs = runs.dropna(subset=[metric_col])
    if runs.empty:
        return None

    best_row = runs.loc[runs[metric_col].idxmin()]
    return best_row.run_id


def main() -> None:
    client = MlflowClient()

    for h in HORIZONS:
        run_id = _best_run_id(h)
        if run_id is None:
            print(
                f"No BEST run found for horizon=+{h}d in experiment '{EXPERIMENT_NAME}'."
            )
            continue

        model_uri = f"runs:/{run_id}/best_model"
        model_name = f"{MODEL_NAME_PREFIX}-h{h}"

        print(f"Registering {model_name} from {model_uri}...")
        mlflow.register_model(model_uri, model_name)

        latest_versions = client.get_latest_versions(model_name)
        latest_version = latest_versions[-1].version

        client.transition_model_version_stage(
            name=model_name,
            version=latest_version,
            stage="Staging",
        )

        print(f"{model_name} version {latest_version} moved to Staging")


if __name__ == "__main__":
    main()
