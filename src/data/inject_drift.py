import pandas as pd
import numpy as np
from pathlib import Path

SHIFT = 400
NOISE = 60
N_ROWS = 30

def inject_drift(path="data/processed/gold_features.csv"):
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    backup = path.replace(".csv", "_backup.csv")
    df.to_csv(backup, index=False)
    print(f"Backup: {backup}")

    cols = ["Close","lag_1","lag_2","lag_3","lag_5","lag_7","lag_10","rolling_mean_7"]
    noise = np.random.normal(0, NOISE, N_ROWS)

    for col in cols:
        if col in df.columns:
            df.loc[df.index[-N_ROWS:], col] += SHIFT + noise

    df.to_csv(path, index=False)
    print(f"Drift injected: +{SHIFT} USD pada {N_ROWS} baris terakhir")
    print(f"Close terakhir setelah drift: {df.iloc[-1]['Close']:.2f}")

if __name__ == "__main__":
    inject_drift()