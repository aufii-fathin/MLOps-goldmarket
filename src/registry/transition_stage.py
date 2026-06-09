import os
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
import mlflow
from mlflow.tracking import MlflowClient

mlflow_tracking_uri = os.getenv(
    "MLFLOW_TRACKING_URI",
    "https://dagshub.com/aufii-fathin/MLOps-goldmarket.mlflow"
)
os.environ.setdefault("MLFLOW_TRACKING_USERNAME", "aufii-fathin")
os.environ.setdefault(
    "MLFLOW_TRACKING_PASSWORD",
    "b0815f5526a09cd2bf06c4324766bd8655aec0ed"
)
mlflow.set_tracking_uri(mlflow_tracking_uri)

EXPERIMENT_NAME = "gold-close-forecast"
HORIZONS = [1, 3, 5, 7]
MODEL_NAME_PREFIX = "gold-close-model"


def _get_best_run_per_model_type(horizon: int):
    """
    Cari run terbaru per model type (sklearn vs ARIMA) untuk horizon tertentu.
    Return dict: {model_type: {run_id, rmse, name}}
    """
    runs = mlflow.search_runs(
        experiment_names=[EXPERIMENT_NAME],
        filter_string=f"tags.mlflow.runName LIKE 'BEST_%_h{horizon}_final'",
        order_by=["start_time DESC"],
    )  

    if runs.empty:
        return {}

    metric_col = None
    for col in ("metrics.avg_rmse", "metrics.rmse"):
        if col in runs.columns:
            metric_col = col
            break

    if metric_col is None:
        return {}

    runs = runs.dropna(subset=[metric_col])
    if runs.empty:
        return {}

    arima_runs = runs[runs["tags.mlflow.runName"].str.contains("ARIMA")]
    sklearn_runs = runs[~runs["tags.mlflow.runName"].str.contains("ARIMA")]

    result = {}

    if not arima_runs.empty:
        latest_arima = arima_runs.iloc[0]
        result["arima"] = {
            "run_id": latest_arima.run_id,
            "rmse": float(latest_arima[metric_col]),
            "name": latest_arima["tags.mlflow.runName"],
        }

    if not sklearn_runs.empty:
        latest_sklearn = sklearn_runs.iloc[0]
        result["sklearn"] = {
            "run_id": latest_sklearn.run_id,
            "rmse": float(latest_sklearn[metric_col]),
            "name": latest_sklearn["tags.mlflow.runName"],
        }

    return result

def _get_production_rmse(client, model_name: str) -> float | None:
    """Ambil avg_rmse model yang sedang production di registry."""
    try:
        mv = client.get_model_version_by_alias(model_name, "production")
        run = mlflow.get_run(mv.run_id)
        for key in ("avg_rmse", "rmse"):
            if key in run.data.metrics:
                return run.data.metrics[key]
    except Exception:
        return None

def main() -> None:
    client = MlflowClient()

    for h in HORIZONS:
        print(f"\n{'='*50}")
        print(f"Processing horizon: +{h}d")
        print(f"{'='*50}")

        candidates = _get_best_run_per_model_type(h)

        if not candidates:
            print(f"  No BEST runs found for horizon=+{h}d, skipping.")
            continue

        model_name = f"{MODEL_NAME_PREFIX}-h{h}"

        # Register semua kandidat
        registered = {}
        for model_type, info in candidates.items():
            model_uri = f"runs:/{info['run_id']}/best_model"
            print(f"  Registering {model_type} (RMSE={info['rmse']:.4f}) from {info['run_id'][:8]}...")

            mv = mlflow.register_model(model_uri, model_name)
            registered[model_type] = {
                "version": mv.version,
                "rmse": info["rmse"],
            }
            print(f"    → Registered as version {mv.version}")

        if "sklearn" in registered:
            sklearn_version = registered["sklearn"]["version"]
            new_rmse = registered["sklearn"]["rmse"]
            current_rmse = _get_production_rmse(client, model_name)

            if current_rmse is None or new_rmse < current_rmse:
                client.set_registered_model_alias(
                    model_name, "production", sklearn_version
                )
                label = f"(sebelumnya {current_rmse:.4f})" if current_rmse else "(pertama kali)"
                print(f"\n  ✅ PROMOTED → v{sklearn_version} RMSE={new_rmse:.4f} {label}")
            else:
                print(f"\n  ⏭️  SKIP → model lama lebih baik "
                    f"(lama={current_rmse:.4f} vs baru={new_rmse:.4f})")

        if "arima" in registered:
            arima_version = registered["arima"]["version"]
            client.set_registered_model_alias(model_name, "staging", arima_version)
            print(f"  📦 STAGING    → arima (v{arima_version}, RMSE={registered['arima']['rmse']:.4f})")

    print(f"\n{'='*50}")
    print("Model registration complete!")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()