import pandas as pd
import yfinance as yf
import joblib
import json
from pathlib import Path


def main():

    horizons = [1, 3, 5, 7]

    model_dir = Path("models")

    # LOAD FEATURE COLUMNS
    feature_cols_path = model_dir / "feature_columns.json"
    if not feature_cols_path.exists():
        raise FileNotFoundError(
            "Missing models/feature_columns.json. Run training first (src/models/train.py)."
        )
    with open(feature_cols_path, "r", encoding="utf-8") as f:
        feature_cols = json.load(f)

    # LOAD MODELS & SCALERS PER HORIZON
    models = {}
    scalers = {}
    for h in horizons:
        model_path = model_dir / f"best_model_h{h}.pkl"
        scaler_path = model_dir / f"scaler_h{h}.pkl"

        # Backward-compatible fallback for horizon=1
        if h == 1 and (not model_path.exists() or not scaler_path.exists()):
            model_path = model_dir / "best_model.pkl"
            scaler_path = model_dir / "scaler.pkl"

        models[h] = joblib.load(model_path)
        scalers[h] = joblib.load(scaler_path)

    # FETCH LATEST DATA
    df = yf.download("GC=F", period="30d", interval="1d")

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.reset_index()

    # FEATURE ENGINEERING
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date")
    df = df.drop_duplicates(subset="Date")

    # Keep only needed base column
    df = df[["Date", "Close"]].copy()

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

    # TAKE LATEST ROW
    latest = df.iloc[-1:]
    latest_ts = pd.to_datetime(latest["Date"].iloc[0])

    X = latest[feature_cols]

    print(
        "\nLatest gold Close:",
        float(latest["Close"].values[0]),
        "| Date:",
        latest_ts.date(),
    )

    for h in horizons:
        X_scaled = scalers[h].transform(X)
        pred_close = float(models[h].predict(X_scaled)[0])
        forecast_date = (latest_ts + pd.Timedelta(days=h)).date()
        print(f"Predicted Close +{h}d ({forecast_date}):", pred_close)


if __name__ == "__main__":
    main()
