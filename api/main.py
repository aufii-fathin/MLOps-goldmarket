from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import mlflow
import mlflow.pyfunc
import pandas as pd
import numpy as np
import yfinance as yf
import joblib
import json
import os
from pathlib import Path

app = FastAPI(title="Gold Market Forecasting API")

mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://mlflow-server:5000"))

MODEL_NAME_PREFIX = "gold-close-model"
HORIZONS = [1, 3, 5, 7]

# Cache models supaya tidak load ulang tiap request
_models = {}
_scalers = {}
_feature_cols = None

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
    try:
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