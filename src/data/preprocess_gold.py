import pandas as pd
from pathlib import Path


def load_raw_data():
    path = Path("data/raw/gold_prices.csv")
    df = pd.read_csv(path)
    return df


def clean_data(df):

    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date")
    df = df.drop_duplicates(subset="Date")

    if "Close" in df.columns:
        df = df[["Date", "Close"]].copy()

    return df


def create_features(df):

    # return (t-1 -> t)
    df["return_1d"] = df["Close"].pct_change()

    # lag features (past Close values)
    df["lag_1"] = df["Close"].shift(1)
    df["lag_2"] = df["Close"].shift(2)
    df["lag_3"] = df["Close"].shift(3)
    df["lag_5"] = df["Close"].shift(5)
    df["lag_7"] = df["Close"].shift(7)
    df["lag_10"] = df["Close"].shift(10)

    # rolling statistics (include up to time t)
    df["rolling_mean_7"] = df["Close"].rolling(7).mean()
    df["rolling_std_7"] = df["Close"].rolling(7).std()

    # volatility (return volatility up to time t)
    df["volatility_20"] = df["return_1d"].rolling(20).std()

    # multi-horizon targets (direct forecasting)
    df["y_1"] = df["Close"].shift(-1)
    df["y_3"] = df["Close"].shift(-3)
    df["y_5"] = df["Close"].shift(-5)
    df["y_7"] = df["Close"].shift(-7)

    df = df.dropna()

    return df


def save_processed(df):

    path = Path("data/processed/gold_features.csv")
    path.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(path, index=False)
    print(f"Processed dataset saved to {path}")


def main():

    df = load_raw_data()
    df = clean_data(df)
    df = create_features(df)
    save_processed(df)


if __name__ == "__main__":
    main()
