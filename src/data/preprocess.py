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

    return df


def create_features(df):

    # return
    df["return"] = df["Close"].pct_change()

    # lag features
    df["lag_1"] = df["Close"].shift(1)
    df["lag_2"] = df["Close"].shift(2)
    df["lag_3"] = df["Close"].shift(3)
    df["lag_5"] = df["Close"].shift(5)
    df["lag_7"] = df["Close"].shift(7)
    df["lag_10"] = df["Close"].shift(10)

    # rolling statistics
    df["rolling_mean_7"] = df["Close"].rolling(7).mean()
    df["rolling_std_7"] = df["Close"].rolling(7).std()

    # volatility
    df["volatility_20"] = df["return"].rolling(20).std()

    # target
    df["target"] = df["Close"].shift(-1)

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