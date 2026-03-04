import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from xgboost import XGBRegressor

def main():
    # LOAD DATA
    df = pd.read_csv("data/processed/gold_features.csv")
    df["Date"] = pd.to_datetime(df["Date"])

    # TARGET (harus sudah dibuat di preprocessing)
    target = "target"

    X = df.drop(columns=["Date", "Close", "target"])
    y = df[target]

    # TIME SERIES SPLIT
    split = int(len(df) * 0.8)

    X_train = X.iloc[:split]
    X_test = X.iloc[split:]

    y_train = y.iloc[:split]
    y_test = y.iloc[split:]

    # FEATURE SCALING
    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    joblib.dump(scaler, "models/scaler.pkl")

    # RESULTS STORAGE
    results = []
    models = {}

    # EVALUATION FUNCTION
    def evaluate(y_true, y_pred, model_name):

        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)

        results.append({
            "model": model_name,
            "rmse": rmse,
            "mae": mae,
            "r2": r2
        })

        print(f"\nModel: {model_name}")
        print("RMSE:", rmse)
        print("MAE :", mae)
        print("R2  :", r2)


    # Linear Regression
    linear = LinearRegression()
    linear.fit(X_train, y_train)
    pred = linear.predict(X_test)
    evaluate(y_test, pred, "Linear Regression")
    models["Linear Regression"] = linear

    # Random Forest
    rf = RandomForestRegressor(
        n_estimators=200,
        max_depth=10,
        random_state=42
    )

    rf.fit(X_train, y_train)
    pred = rf.predict(X_test)
    evaluate(y_test, pred, "Random Forest")
    models["Random Forest"] = rf

    # Gradient Boosting
    gbr = GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=3,
        random_state=42
    )

    gbr.fit(X_train, y_train)
    pred = gbr.predict(X_test)
    evaluate(y_test, pred, "Gradient Boosting")
    models["Gradient Boosting"] = gbr

    # XGBoost
    xgb = XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )

    xgb.fit(X_train, y_train)
    pred = xgb.predict(X_test)
    evaluate(y_test, pred, "XGBoost")
    models["XGBoost"] = xgb

    # SELECT BEST MODEL
    results_df = pd.DataFrame(results)
    best_model_name = results_df.sort_values("rmse").iloc[0]["model"]

    print("\n")
    print("Best Model:", best_model_name)

    best_model = models[best_model_name]

    # SAVE BEST MODEL
    model_path = Path("models")
    model_path.mkdir(exist_ok=True)

    save_path = model_path / "best_model.pkl"
    joblib.dump(best_model, save_path)
    print(f"Best model saved to {save_path}")

    # SAVE RESULTS TABLE
    results_df.to_csv("models/model_results.csv", index=False)
    print("\nModel comparison saved to models/model_results.csv")

if __name__ == "__main__":
    main()