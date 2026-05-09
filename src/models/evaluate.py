import pandas as pd
import sys

THRESHOLD_RMSE = 0.02

results = pd.read_csv("models/model_results.csv")

best_rmse = results["rmse"].min()

print(f"Best RMSE: {best_rmse:.6f}")

if best_rmse <= THRESHOLD_RMSE:
    print("Model passed validation")
else:
    print("Model failed validation")
    sys.exit(1)