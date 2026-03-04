import yfinance as yf
import pandas as pd
from pathlib import Path


def fetch_gold_data(period="10y"):
    """
    Fetch historical gold price data from Yahoo Finance.
    """
    ticker = "GC=F"

    df = yf.download(
        ticker,
        period=period,
        interval="1d"
    )

    # handle multi-index columns
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.reset_index()

    return df


def save_raw_data(df):
    """
    Save dataset to data/raw directory.
    """
    output_path = Path("data/raw/gold_prices.csv")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(output_path, index=False)

    print(f"Dataset saved to {output_path}")


def main():
    df = fetch_gold_data()

    print("Data fetched successfully")
    print(df.head())

    save_raw_data(df)


if __name__ == "__main__":
    main()