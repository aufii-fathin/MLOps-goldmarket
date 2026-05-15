from fastapi import FastAPI
import mlflow
import os

app = FastAPI()

mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://mlflow-server:5000"))

@app.get("/")
def home():
    return {"message": "Gold Market API Running"}

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.get("/test-mlflow")
def test_mlflow():
    with mlflow.start_run():
        mlflow.log_param("test_param", "hello")
        mlflow.log_metric("test_metric", 1.0)
    return {"message": "MLflow logging successful"}