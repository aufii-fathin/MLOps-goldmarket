import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
import mlflow

mlflow.set_tracking_uri("sqlite:///mlflow.db")
from mlflow.tracking import MlflowClient

EXPERIMENT_NAME = "gold-close-forecast"
HORIZONS = [1, 3, 5, 7]
MODEL_NAME_PREFIX = "gold-close-model"


def _latest_best_run_id(horizon: int) -> str | None:
    runs = mlflow.search_runs(
        experiment_names=[EXPERIMENT_NAME],
        filter_string=f"tags.mlflow.runName LIKE 'BEST_%_h{horizon}_final'",
        order_by=["start_time DESC"],
        max_results=1,
    )
    if runs.empty:
        return None
    return runs.iloc[0].run_id


def main() -> None:
    client = MlflowClient()

    for h in HORIZONS:
        run_id = _latest_best_run_id(h)
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
