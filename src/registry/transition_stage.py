import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import mlflow
mlflow.set_tracking_uri("sqlite:///mlflow.db")
from mlflow.tracking import MlflowClient

MODEL_NAME = "gold-price-model"

# Find the best model run
runs = mlflow.search_runs(experiment_names=["gold-price-prediction"], filter_string="tags.mlflow.runName LIKE 'BEST_%_final'")
if runs.empty:
    raise ValueError("No best model run found")
run_id = runs.iloc[0].run_id
model_uri = f"runs:/{run_id}/best_model"

# Register the model
mlflow.register_model(model_uri, MODEL_NAME)

# Now transition to staging
client = MlflowClient()
latest_versions = client.get_latest_versions(MODEL_NAME)
latest_version = latest_versions[-1].version

client.transition_model_version_stage(
    name=MODEL_NAME,
    version=latest_version,
    stage="Staging"
)

print(f"Version {latest_version} moved to Staging")