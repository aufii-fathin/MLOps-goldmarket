import pandas as pd
import numpy as np
from scipy import stats
import json, sys
from pathlib import Path

DRIFT_THRESHOLD = 0.05
FEATURES_TO_CHECK = ["Close", "return_1d", "rolling_mean_7", "volatility_20"]
features, drift_detected = {}, False

def detect_drift(path="data/processed/gold_features.csv", window=30):
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date")

    cutoff = df["Date"].max() - pd.Timedelta(days=window)
    baseline = df[df["Date"] < cutoff]
    recent   = df[df["Date"] >= cutoff]

    if len(recent) < 5:
        result = {"drift_detected": False, "reason": "insufficient_data"}
        print(json.dumps(result, indent=2))
        return result

    features, drift_detected = {}, False
    for feat in FEATURES_TO_CHECK:
        if feat not in df.columns:
            continue
        ks, p = stats.ks_2samp(baseline[feat].dropna(), recent[feat].dropna())
        drifted = bool(p < DRIFT_THRESHOLD)
        if drifted:
            drift_detected = True
        features[feat] = {"ks_statistic": round(float(ks), 4),
                          "p_value": round(float(p), 4),
                          "drift": drifted}
        status = "⚠️  DRIFT" if drifted else "✅ OK"
        print(f"  {feat:<20} KS={ks:.4f}  p={p:.4f}  {status}")

    summary = {"drift_detected": bool(drift_detected),
                "recent_rows": int(len(recent)),
                "baseline_rows": int(len(baseline)),
                "features": features}

    Path("models").mkdir(exist_ok=True)
    with open("models/drift_report.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nDrift detected: {drift_detected}")
    return summary

if __name__ == "__main__":
    detect_drift()