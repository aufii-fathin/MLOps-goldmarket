import pandas as pd
import sys

THRESHOLD_RMSE_PCT = 0.02

results = pd.read_csv("models/model_results.csv")

if "rmse_pct" in results.columns:
    best_rmse_pct = results["rmse_pct"].min()
    print(f"Best RMSE%: {best_rmse_pct * 100:.3f}%")

    if best_rmse_pct <= THRESHOLD_RMSE_PCT:
        print("Model passed validation")
    else:
        print("Model failed validation")
        sys.exit(1)
else:
    best_rmse = results["rmse"].min()
    print(f"Best RMSE: {best_rmse:.6f}")
    print("Validation skipped: rmse_pct not found in results.")
