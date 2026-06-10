from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, FileResponse
import mlflow
import mlflow.pyfunc
import mlflow.tracking
import pandas as pd
import numpy as np
import yfinance as yf
import joblib
import json
import os
from datetime import datetime
from pathlib import Path
from prometheus_client import (
    Counter,
    Histogram,
    Gauge,
    generate_latest,
    CONTENT_TYPE_LATEST,
)
import psutil
import time
from datetime import timezone, timedelta

app = FastAPI(title="Gold Market Forecasting API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET"],
    allow_headers=["*"],
)

os.environ.setdefault("MLFLOW_TRACKING_USERNAME", "aufii-fathin")
os.environ.setdefault(
    "MLFLOW_TRACKING_PASSWORD",
    "b0815f5526a09cd2bf06c4324766bd8655aec0ed",
)
mlflow.set_tracking_uri(
    os.getenv(
        "MLFLOW_TRACKING_URI",
        "https://dagshub.com/aufii-fathin/MLOps-goldmarket.mlflow",
    )
)

MODEL_NAME_PREFIX = "gold-close-model"
HORIZONS = [1, 3, 5, 7]

REQUEST_COUNT = Counter("request_count_total", "Total Prediction Requests")
INFERENCE_LATENCY = Histogram("inference_latency_seconds", "Prediction Latency")
CPU_USAGE = Gauge("cpu_usage_percent", "CPU Usage Percent")
MEMORY_USAGE = Gauge("memory_usage_percent", "Memory Usage Percent")
PREDICTION_H1 = Gauge("prediction_h1", "Latest H1 Prediction")

DRIFT_DETECTED = Gauge(
    "gold_drift_detected", "1 jika data drift terdeteksi, 0 jika tidak"
)

PRODUCTION_RMSE = {
    h: Gauge(f"gold_production_rmse_h{h}", f"RMSE model production horizon {h}")
    for h in HORIZONS
}

_models: dict = {}
_scalers: dict = {}
_feature_cols: list | None = None

def _load_resources():
    global _feature_cols
    if _feature_cols is None:
        with open("models/feature_columns.json") as f:
            _feature_cols = json.load(f)

    for h in HORIZONS:
        if h not in _models:
            model_uri = f"models:/{MODEL_NAME_PREFIX}-h{h}@production"
            _models[h] = mlflow.pyfunc.load_model(model_uri)
            _scalers[h] = joblib.load(f"models/scaler_h{h}.pkl")

def _build_features() -> pd.DataFrame:
    df = yf.download("GC=F", period="60d", interval="1d", progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.reset_index()[["Date", "Close"]].dropna()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date")

    df["return_1d"] = df["Close"].pct_change()
    df["lag_1"] = df["Close"].shift(1)
    df["lag_2"] = df["Close"].shift(2)
    df["lag_3"] = df["Close"].shift(3)
    df["lag_5"] = df["Close"].shift(5)
    df["lag_7"] = df["Close"].shift(7)
    df["lag_10"] = df["Close"].shift(10)
    df["rolling_mean_7"] = df["Close"].rolling(7).mean()
    df["rolling_std_7"] = df["Close"].rolling(7).std()
    df["volatility_20"] = df["return_1d"].rolling(20).std()
    df = df.dropna()

    return df

@app.get("/")
def home():
    return {"message": "Gold Market Forecasting API Running"}


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.get("/test-mlflow")
def test_mlflow():
    with mlflow.start_run():
        mlflow.log_param("test_param", "hello")
        mlflow.log_metric("test_metric", 1.0)
    return {"message": "MLflow logging successful"}


@app.get("/predict")
def predict():
    start_time = time.time()
    try:
        REQUEST_COUNT.inc()
        _load_resources()
        df = _build_features()
        latest = df.iloc[-1:]
        latest_date = pd.to_datetime(latest["Date"].iloc[0])
        current_price = float(latest["Close"].iloc[0])

        X = latest[_feature_cols]
        predictions = {}

        for h in HORIZONS:
            X_scaled = _scalers[h].transform(X)
            X_scaled_df = pd.DataFrame(X_scaled, columns=_feature_cols)
            pred = float(_models[h].predict(X_scaled_df)[0])
            forecast_date = (latest_date + pd.Timedelta(days=h)).date()
            predictions[f"h{h}"] = {
                "horizon_days": h,
                "forecast_date": str(forecast_date),
                "predicted_close": round(pred, 2),
            }
            if h == 1:
                PREDICTION_H1.set(pred)

        INFERENCE_LATENCY.observe(time.time() - start_time)
        CPU_USAGE.set(psutil.cpu_percent())
        MEMORY_USAGE.set(psutil.virtual_memory().percent)

        return {
            "current_date": str(latest_date.date()),
            "current_price": round(current_price, 2),
            "model": "Linear Regression @production",
            "predictions": predictions,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/models/info")
def models_info():
    info = {}
    for h in HORIZONS:
        name = f"{MODEL_NAME_PREFIX}-h{h}"
        info[f"h{h}"] = {
            "model_name": name,
            "production_alias": f"models:/{name}@production",
            "staging_alias": f"models:/{name}@staging",
        }
    return info


@app.get("/history")
def history():
    """Mengembalikan 30 hari terakhir harga penutupan gold (XAU/USD)."""
    df = yf.download("GC=F", period="35d", interval="1d", progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.reset_index()[["Date", "Close"]].dropna().tail(30)
    return [
        {"date": str(row["Date"].date()), "close": round(float(row["Close"]), 2)}
        for _, row in df.iterrows()
    ]

@app.get("/system/info")
def system_info():
    """Metadata sistem: data terbaru, last trained, trigger reason, drift status."""

    try:
        df = pd.read_csv("data/processed/gold_features.csv")
        last_data_date = str(pd.to_datetime(df["Date"]).max().date())
    except Exception:
        last_data_date = "unknown"

    last_trained_at = "unknown"
    trigger_reason = "unknown"
    model_version = "unknown"
    try:
        client = mlflow.tracking.MlflowClient()
        experiment = client.get_experiment_by_name("gold-close-forecast")
        if experiment:
            runs = client.search_runs(
                experiment_ids=[experiment.experiment_id],
                filter_string="tags.mlflow.runName LIKE 'BEST_%_final'",
                order_by=["start_time DESC"],
                max_results=1,
            )
            if runs:
                run = runs[0]
                WIB = timezone(timedelta(hours=7))
                last_trained_at = datetime.fromtimestamp(
                    run.info.start_time / 1000, tz=timezone.utc
                ).astimezone(WIB).strftime("%d %b %Y, %H:%M WIB")
                trigger_reason = run.data.tags.get("trigger", "scheduled")
                model_version = run.data.tags.get(
                    "mlflow.source.git.commit", ""
                )[:7] or "unknown"
    except Exception:
        pass

    drift_detected = False
    drift_features: dict = {}
    try:
        with open("models/drift_report.json") as f:
            report = json.load(f)
        drift_detected = bool(report.get("drift_detected", False))
        drift_features = {
            k: v for k, v in report.get("features", {}).items() if v.get("drift")
        }
    except Exception:
        pass

    return {
        "last_data_date": last_data_date,
        "last_trained_at": last_trained_at,
        "trigger_reason": trigger_reason,
        "model_version": model_version,
        "drift_detected": drift_detected,
        "drifted_features": list(drift_features.keys()),
    }

@app.get("/metrics")
def metrics():
    # Drift gauge
    try:
        with open("models/drift_report.json") as f:
            report = json.load(f)
        DRIFT_DETECTED.set(1 if report.get("drift_detected") else 0)
    except FileNotFoundError:
        DRIFT_DETECTED.set(0)

    # Production RMSE per horizon (best model = min RMSE di CSV)
    for h in HORIZONS:
        try:
            df = pd.read_csv(f"models/model_results_h{h}.csv")
            PRODUCTION_RMSE[h].set(float(df["rmse"].min()))
        except Exception:
            PRODUCTION_RMSE[h].set(0)

    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.get("/dashboard", include_in_schema=False)
def dashboard():
    html_path = Path("web/dashboard.html")
    if not html_path.exists():
        raise HTTPException(status_code=404, detail="dashboard.html not found in web/")
    return FileResponse(html_path)