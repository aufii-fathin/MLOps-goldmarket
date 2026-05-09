import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from xgboost import XGBRegressor

import mlflow
import mlflow.sklearn
import mlflow.xgboost

import warnings
import logging
logging.getLogger("mlflow").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=UserWarning, module="mlflow")

def main():
    # LOAD DATA
    df = pd.read_csv("data/processed/gold_features.csv")
    df["Date"] = pd.to_datetime(df["Date"])

    # TARGET
    target = "target"

    X = df.drop(columns=["Date", "Close", "target"])
    y = df[target]

    # TIME SERIES SPLIT (walk-forward, 5 folds)
    tscv = TimeSeriesSplit(n_splits=5)

    # MODEL DEFINITIONS + PARAMS
    model_defs = {
        "Linear Regression": {
            "model": LinearRegression(fit_intercept=False),
            "params": {
                "fit_intercept": False
            }
        },
        "Random Forest": {
            "model": RandomForestRegressor(
                n_estimators=200,
                max_depth=10,
                random_state=42
            ),
            "params": {
                "n_estimators": 200,
                "max_depth": 10,
                "random_state": 42
            }
        },
        "Gradient Boosting": {
            "model": GradientBoostingRegressor(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=3,
                random_state=42
            ),
            "params": {
                "n_estimators": 200,
                "learning_rate": 0.05,
                "max_depth": 3,
                "random_state": 42
            }
        },
        "XGBoost": {
            "model": XGBRegressor(
                n_estimators=300,
                learning_rate=0.05,
                max_depth=5,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42
            ),
            "params": {
                "n_estimators": 300,
                "learning_rate": 0.05,
                "max_depth": 5,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "random_state": 42
            }
        },
    }

    # RESULTS STORAGE (per fold)
    fold_results = {name: {"rmse": [], "mae": [], "r2": []} for name in model_defs}

    # MLflow experiment
    mlflow.set_experiment("gold-price-prediction")

    # WALK-FORWARD VALIDATION (per model per fold = 1 MLflow run)
    for name, config in model_defs.items():
        model = config["model"]
        params = config["params"]

        print(f"\n{'='*40}")
        print(f"Training: {name}")

        for fold, (train_idx, test_idx) in enumerate(tscv.split(X), start=1):
            print(f"\n  Fold {fold}")
            print(f"    Train: {df['Date'].iloc[train_idx[0]].date()} -> {df['Date'].iloc[train_idx[-1]].date()}")
            print(f"    Test : {df['Date'].iloc[test_idx[0]].date()} -> {df['Date'].iloc[test_idx[-1]].date()}")

            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            scaler = StandardScaler()
            X_train_sc = scaler.fit_transform(X_train)
            X_test_sc = scaler.transform(X_test)

            with mlflow.start_run(run_name=f"{name}_fold{fold}"):

                # Log params
                mlflow.log_param("model", name)
                mlflow.log_param("fold", fold)
                mlflow.log_param("train_size", len(train_idx))
                mlflow.log_param("test_size", len(test_idx))
                for k, v in params.items():
                    mlflow.log_param(k, v)

                # Train
                if name == "Linear Regression":
                    model.fit(X_train_sc, y_train)
                    pred = model.predict(X_test_sc)
                else:
                    model.fit(X_train, y_train)
                    pred = model.predict(X_test)

                # Metrics
                rmse = np.sqrt(mean_squared_error(y_test, pred))
                mae = mean_absolute_error(y_test, pred)
                r2 = r2_score(y_test, pred)

                mlflow.log_metric("rmse", rmse)
                mlflow.log_metric("mae", mae)
                mlflow.log_metric("r2", r2)

                fold_results[name]["rmse"].append(rmse)
                fold_results[name]["mae"].append(mae)
                fold_results[name]["r2"].append(r2)

                print(f"    RMSE: {rmse:.6f} | MAE: {mae:.6f} | R2: {r2:.4f}")

                # Log model per fold
                if name == "XGBoost":
                    mlflow.xgboost.log_model(model, name="model")
                else:
                    mlflow.sklearn.log_model(model, name="model")

    # AVERAGE METRICS ACROSS FOLDS
    print(f"\n{'='*40}")
    print("AVERAGE METRICS ACROSS FOLDS")
    print(f"{'='*40}")

    results = []
    for name, metrics in fold_results.items():
        avg_rmse = np.mean(metrics["rmse"])
        avg_mae = np.mean(metrics["mae"])
        avg_r2 = np.mean(metrics["r2"])

        results.append({
            "model": name,
            "rmse": avg_rmse,
            "mae": avg_mae,
            "r2": avg_r2
        })

        print(f"\nModel: {name}")
        print(f"  RMSE: {avg_rmse:.6f}")
        print(f"  MAE : {avg_mae:.6f}")
        print(f"  R2  : {avg_r2:.4f}")

    results_df = pd.DataFrame(results)

    # SELECT BEST MODEL (by avg RMSE)
    best_model_name = results_df.sort_values("rmse").iloc[0]["model"]
    print(f"\nBest Model: {best_model_name}")

    # RETRAIN BEST MODEL ON FULL DATA
    scaler_final = StandardScaler()
    X_all = scaler_final.fit_transform(X)

    best_model = model_defs[best_model_name]["model"]
    best_params = model_defs[best_model_name]["params"]
    best_model.fit(X_all, y)

    # LOG BEST MODEL KE MLFLOW
    with mlflow.start_run(run_name=f"BEST_{best_model_name}_final"):
        mlflow.log_param("model", best_model_name)
        mlflow.log_param("retrained_on", "full_data")
        for k, v in best_params.items():
            mlflow.log_param(k, v)

        best_metrics = results_df[results_df["model"] == best_model_name].iloc[0]
        mlflow.log_metric("avg_rmse", best_metrics["rmse"])
        mlflow.log_metric("avg_mae", best_metrics["mae"])
        mlflow.log_metric("avg_r2", best_metrics["r2"])

        if best_model_name == "XGBoost":
            mlflow.xgboost.log_model(best_model, name="best_model")
        else:
            mlflow.sklearn.log_model(best_model, name="best_model")

    # SAVE LOCALLY
    model_path = Path("models")
    model_path.mkdir(exist_ok=True)

    joblib.dump(best_model, model_path / "best_model.pkl")
    joblib.dump(scaler_final, model_path / "scaler.pkl")
    results_df.to_csv(model_path / "model_results.csv", index=False)

    print(f"Best model saved to models/best_model.pkl")
    print(f"Scaler saved to models/scaler.pkl")
    print(f"Results saved to models/model_results.csv")


if __name__ == "__main__":
    main()