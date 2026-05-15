import pandas as pd
import sys

THRESHOLD_RMSE_BY_HORIZON = {
    1: 30.0,
    3: 52.0,
    5: 66.0,
    7: 78.0,
}


def _load_ml_results(horizon: int) -> pd.DataFrame | None:
    try:
        path = f"models/model_results_h{horizon}.csv"
        return pd.read_csv(path)
    except FileNotFoundError:
        if horizon == 1:
            try:
                return pd.read_csv("models/model_results.csv")
            except FileNotFoundError:
                return None
        return None


def _load_arima_results(horizon: int) -> pd.DataFrame | None:
    try:
        results = pd.read_csv("models/arima_results.csv")
    except FileNotFoundError:
        return None

    if "horizon" in results.columns:
        results = results[results["horizon"] == horizon]
    return results if not results.empty else None


def _best_metrics(results: pd.DataFrame) -> dict:
    return {
        "rmse": float(results["rmse"].min()),
        "rmse_pct": float(results["rmse_pct"].min())
        if "rmse_pct" in results.columns
        else None,
    }


any_failed = False

for horizon, threshold in THRESHOLD_RMSE_BY_HORIZON.items():
    candidates = []

    ml_results = _load_ml_results(horizon)
    if ml_results is not None:
        metrics = _best_metrics(ml_results)
        candidates.append({"source": "ml", **metrics})

    arima_results = _load_arima_results(horizon)
    if arima_results is not None:
        metrics = _best_metrics(arima_results)
        candidates.append({"source": "arima", **metrics})

    if not candidates:
        print(f"No results found for horizon=+{horizon}d")
        any_failed = True
        continue

    best = min(candidates, key=lambda item: item["rmse"])
    print(
        f"Best RMSE h=+{horizon}d ({best['source']}): {best['rmse']:.6f} | "
        f"threshold: {threshold:.2f}"
    )
    if best["rmse_pct"] is not None:
        print(
            f"Best RMSE% h=+{horizon}d ({best['source']}): {best['rmse_pct'] * 100:.3f}%"
        )

    if best["rmse"] > threshold:
        any_failed = True

if any_failed:
    print("Model failed validation")
    sys.exit(1)

print("Model passed validation")
