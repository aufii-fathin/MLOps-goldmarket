import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

from mlflow.tracking import MlflowClient

MODEL_NAME = "gold-price-model"

client = MlflowClient()

latest_versions = client.get_latest_versions(MODEL_NAME)

latest_version = latest_versions[-1].version

client.transition_model_version_stage(
    name=MODEL_NAME,
    version=latest_version,
    stage="Staging"
)

print(f"Version {latest_version} moved to Staging")