import os
import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.base import clone

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from xgboost import XGBRegressor

import mlflow
import mlflow.sklearn
import mlflow.xgboost

# Configure MLflow tracking and artifacts for local and CI use
mlflow_tracking_uri = os.getenv(
    "MLFLOW_TRACKING_URI",
    "https://dagshub.com/aufii-fathin/MLOps-goldmarket.mlflow"
)
os.environ.setdefault("MLFLOW_TRACKING_USERNAME", "aufii-fathin")
os.environ.setdefault(
    "MLFLOW_TRACKING_PASSWORD",
    "b0815f5526a09cd2bf06c4324766bd8655aec0ed"
)
# mlflow_artifact_root = os.getenv("MLFLOW_ARTIFACT_ROOT", str(Path.cwd() / "mlruns"))
# Path(mlflow_artifact_root).mkdir(parents=True, exist_ok=True)
# os.environ.setdefault("MLFLOW_ARTIFACT_ROOT", str(Path(mlflow_artifact_root).resolve()))
mlflow.set_tracking_uri(mlflow_tracking_uri)

import warnings
import logging

logging.getLogger("mlflow").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=UserWarning, module="mlflow")


def main():
    # LOAD DATA
    df = pd.read_csv("data/processed/gold_features.csv")
    df["Date"] = pd.to_datetime(df["Date"])

    # MULTI-HORIZON TARGETS (Close forecasting)
    horizons = [1, 3, 5, 7]
    target_cols = [f"y_{h}" for h in horizons]
    missing_targets = [c for c in target_cols if c not in df.columns]
    if missing_targets:
        raise ValueError(
            "Missing target columns in data/processed/gold_features.csv: "
            f"{missing_targets}. Run src/data/preprocess_gold.py first."
        )

    # Features
    X = df.drop(columns=["Date"] + target_cols)
    feature_cols = list(X.columns)

    # TIME SERIES SPLIT (walk-forward, 5 folds)
    tscv = TimeSeriesSplit(n_splits=5, gap=max(horizons))

    # MODEL DEFINITIONS + PARAMS
    model_defs = {
        "Linear Regression": {
            "model": LinearRegression(fit_intercept=True),
            "params": {"fit_intercept": True},
        },
        "Random Forest": {
            "model": RandomForestRegressor(
                n_estimators=200, max_depth=10, random_state=42
            ),
            "params": {"n_estimators": 200, "max_depth": 10, "random_state": 42},
        },
        "Gradient Boosting": {
            "model": GradientBoostingRegressor(
                n_estimators=200, learning_rate=0.05, max_depth=3, random_state=42
            ),
            "params": {
                "n_estimators": 200,
                "learning_rate": 0.05,
                "max_depth": 3,
                "random_state": 42,
            },
        },
        "XGBoost": {
            "model": XGBRegressor(
                n_estimators=300,
                learning_rate=0.05,
                max_depth=5,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
            ),
            "params": {
                "n_estimators": 300,
                "learning_rate": 0.05,
                "max_depth": 5,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "random_state": 42,
            },
        },
    }

    # RESULTS STORAGE: fold_results[horizon][model] -> metrics
    fold_results = {
        h: {
            name: {
                "rmse": [],
                "mae": [],
                "r2": [],
                "rmse_pct": [],
                "naive_rmse": [],
                "naive_mae": [],
                "naive_r2": [],
                "naive_rmse_pct": [],
            }
            for name in model_defs
        }
        for h in horizons
    }

    # MLflow experiment
    mlflow.set_experiment("gold-close-forecast")

    # Persist feature column order for inference
    model_dir = Path("models")
    model_dir.mkdir(exist_ok=True)
    with open(model_dir / "feature_columns.json", "w", encoding="utf-8") as f:
        json.dump(feature_cols, f, indent=2)

    # WALK-FORWARD VALIDATION (per horizon, per model, per fold = 1 MLflow run)
    for horizon in horizons:
        y = df[f"y_{horizon}"]

        print(f"\n{'=' * 60}")
        print(f"Horizon: +{horizon} day(s) (predict Close)")
        print(f"{'=' * 60}")

        for name, config in model_defs.items():
            base_model = config["model"]
            params = config["params"]

            print(f"\n{'=' * 40}")
            print(f"Training: {name}")

            for fold, (train_idx, test_idx) in enumerate(tscv.split(X), start=1):
                print(f"\n  Fold {fold}")
                print(
                    f"    Train: {df['Date'].iloc[train_idx[0]].date()} -> {df['Date'].iloc[train_idx[-1]].date()}"
                )
                print(
                    f"    Test : {df['Date'].iloc[test_idx[0]].date()} -> {df['Date'].iloc[test_idx[-1]].date()}"
                )

                X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

                naive_pred = X_test["Close"].to_numpy()

                scaler = StandardScaler()
                X_train_sc = scaler.fit_transform(X_train)
                X_test_sc = scaler.transform(X_test)

                model = clone(base_model)

                with mlflow.start_run(run_name=f"{name}_h{horizon}_fold{fold}"):
                    # Log params
                    mlflow.log_param("model", name)
                    mlflow.log_param("horizon", horizon)
                    mlflow.log_param("fold", fold)
                    mlflow.log_param("train_size", len(train_idx))
                    mlflow.log_param("test_size", len(test_idx))
                    for k, v in params.items():
                        mlflow.log_param(k, v)

                    # Train (always on scaled features for consistency)
                    model.fit(X_train_sc, y_train)
                    pred = model.predict(X_test_sc)

                    # Metrics
                    rmse = np.sqrt(mean_squared_error(y_test, pred))
                    mae = mean_absolute_error(y_test, pred)
                    r2 = r2_score(y_test, pred)

                    naive_rmse = np.sqrt(mean_squared_error(y_test, naive_pred))
                    naive_mae = mean_absolute_error(y_test, naive_pred)
                    naive_r2 = r2_score(y_test, naive_pred)

                    # Relative RMSE (percentage of average true Close in the fold)
                    y_mean = float(np.mean(y_test))
                    rmse_pct = float(rmse / y_mean) if y_mean != 0 else float("nan")
                    naive_rmse_pct = (
                        float(naive_rmse / y_mean) if y_mean != 0 else float("nan")
                    )

                    mlflow.log_metric("rmse", rmse)
                    mlflow.log_metric("mae", mae)
                    mlflow.log_metric("r2", r2)

                    mlflow.log_metric("naive_rmse", naive_rmse)
                    mlflow.log_metric("naive_mae", naive_mae)
                    mlflow.log_metric("naive_r2", naive_r2)

                    mlflow.log_metric("rmse_pct", rmse_pct)
                    mlflow.log_metric("naive_rmse_pct", naive_rmse_pct)

                    fold_results[horizon][name]["rmse"].append(rmse)
                    fold_results[horizon][name]["mae"].append(mae)
                    fold_results[horizon][name]["r2"].append(r2)

                    fold_results[horizon][name]["rmse_pct"].append(rmse_pct)
                    fold_results[horizon][name]["naive_rmse"].append(naive_rmse)
                    fold_results[horizon][name]["naive_mae"].append(naive_mae)
                    fold_results[horizon][name]["naive_r2"].append(naive_r2)
                    fold_results[horizon][name]["naive_rmse_pct"].append(naive_rmse_pct)

                    print(f"    RMSE: {rmse:.6f} | MAE: {mae:.6f} | R2: {r2:.4f}")
                    print(
                        f"    Naive RMSE: {naive_rmse:.6f} | Naive MAE: {naive_mae:.6f} | Naive R2: {naive_r2:.4f}"
                    )
                    print(
                        f"    RMSE%: {rmse_pct * 100:.3f}% | Naive RMSE%: {naive_rmse_pct * 100:.3f}%"
                    )

                    # # Log model per fold
                    # if name == "XGBoost":
                    #     mlflow.xgboost.log_model(model, artifact_path="model")
                    # else:
                    #     mlflow.sklearn.log_model(model, artifact_path="model")

    # AVERAGE METRICS ACROSS FOLDS + RETRAIN BEST PER HORIZON
    for horizon in horizons:
        print(f"\n{'=' * 40}")
        print(f"AVERAGE METRICS ACROSS FOLDS (horizon=+{horizon}d)")
        print(f"{'=' * 40}")

        results = []
        for name, metrics in fold_results[horizon].items():
            avg_rmse = float(np.mean(metrics["rmse"]))
            avg_mae = float(np.mean(metrics["mae"]))
            avg_r2 = float(np.mean(metrics["r2"]))
            avg_rmse_pct = float(np.mean(metrics["rmse_pct"]))

            avg_naive_rmse = float(np.mean(metrics["naive_rmse"]))
            avg_naive_mae = float(np.mean(metrics["naive_mae"]))
            avg_naive_r2 = float(np.mean(metrics["naive_r2"]))
            avg_naive_rmse_pct = float(np.mean(metrics["naive_rmse_pct"]))

            results.append(
                {
                    "horizon": horizon,
                    "model": name,
                    "rmse": avg_rmse,
                    "mae": avg_mae,
                    "r2": avg_r2,
                    "rmse_pct": avg_rmse_pct,
                    "naive_rmse": avg_naive_rmse,
                    "naive_mae": avg_naive_mae,
                    "naive_r2": avg_naive_r2,
                    "naive_rmse_pct": avg_naive_rmse_pct,
                }
            )

            print(f"\nModel: {name}")
            print(f"  RMSE: {avg_rmse:.6f}")
            print(f"  MAE : {avg_mae:.6f}")
            print(f"  R2  : {avg_r2:.4f}")
            print(f"  RMSE%: {avg_rmse_pct * 100:.3f}%")
            print(f"  Naive RMSE%: {avg_naive_rmse_pct * 100:.3f}%")

        results_df = pd.DataFrame(results).sort_values("rmse")

        best_model_name = results_df.iloc[0]["model"]
        print(f"\nBest Model (h=+{horizon}d): {best_model_name}")

        # Retrain best model on full data for this horizon
        y_full = df[f"y_{horizon}"]
        scaler_final = StandardScaler()
        X_all = scaler_final.fit_transform(X)

        best_model = clone(model_defs[best_model_name]["model"])
        best_params = model_defs[best_model_name]["params"]
        best_model.fit(X_all, y_full)

        # MLflow run for final model
        with mlflow.start_run(run_name=f"BEST_{best_model_name}_h{horizon}_final"):
            mlflow.log_param("model", best_model_name)
            mlflow.log_param("horizon", horizon)
            mlflow.log_param("retrained_on", "full_data")
            for k, v in best_params.items():
                mlflow.log_param(k, v)

            best_metrics = results_df.iloc[0]
            mlflow.log_metric("avg_rmse", best_metrics["rmse"])
            mlflow.log_metric("avg_mae", best_metrics["mae"])
            mlflow.log_metric("avg_r2", best_metrics["r2"])

            if best_model_name == "XGBoost":
                mlflow.xgboost.log_model(best_model, artifact_path="best_model")
            else:
                mlflow.sklearn.log_model(best_model, artifact_path="best_model")

        # Save per-horizon artifacts
        joblib.dump(best_model, model_dir / f"best_model_h{horizon}.pkl")
        joblib.dump(scaler_final, model_dir / f"scaler_h{horizon}.pkl")
        results_df.to_csv(model_dir / f"model_results_h{horizon}.csv", index=False)

        if horizon == 1:
            joblib.dump(best_model, model_dir / "best_model.pkl")
            joblib.dump(scaler_final, model_dir / "scaler.pkl")
            results_df.drop(columns=["horizon"]).to_csv(
                model_dir / "model_results.csv", index=False
            )

    print("\nTraining complete. Artifacts saved under models/.")


if __name__ == "__main__":
    main()
