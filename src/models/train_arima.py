import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit

warnings.filterwarnings("ignore")


def _get_mlflow():
    try:
        import mlflow

        mlflow.set_tracking_uri("sqlite:///mlflow_new.db")
        return mlflow
    except Exception:
        return None


@dataclass(frozen=True)
class FoldMetrics:
    rmse: float
    mae: float
    r2: float
    rmse_pct: float
    naive_rmse: float
    naive_mae: float
    naive_r2: float
    naive_rmse_pct: float


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def _rmse_pct(y_true: np.ndarray, rmse_value: float) -> float:
    denom = float(np.mean(y_true))
    return float(rmse_value / denom) if denom != 0 else float("nan")


def _fit_arima(close_train: np.ndarray, order: tuple[int, int, int]):
    try:
        from statsmodels.tsa.arima.model import ARIMA
    except Exception as exc:  # pragma: no cover
        raise ImportError(
            "statsmodels is required for ARIMA. Install with: pip install statsmodels"
        ) from exc

    model = ARIMA(
        close_train,
        order=order,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    return model.fit()


def _fit_garch(residuals: np.ndarray):
    try:
        from arch import arch_model
    except Exception as exc:
        raise ImportError(
            "arch is required for GARCH. Install with: pip install arch"
        ) from exc

    am = arch_model(residuals, mean="Zero", vol="GARCH", p=1, q=1, dist="normal")
    return am.fit(disp="off")


def _forecast_arima_mean(arima_fit, steps: int) -> np.ndarray:
    return np.asarray(arima_fit.forecast(steps=steps), dtype=float)


def _predict_for_fold(
    df: pd.DataFrame,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    horizon: int,
    arima_order: tuple[int, int, int],
    use_garch: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    train_end = int(train_idx[-1])
    max_h = 7

    close_series = df["Close"].to_numpy(dtype=float)
    close_train = close_series[train_idx]

    arima_fit = _fit_arima(close_train, arima_order)

    y_true = df[f"y_{horizon}"].iloc[test_idx].to_numpy(dtype=float)

    preds: list[float] = []
    sigmas: list[float] = []

    garch_sigma = None
    if use_garch:
        resid = np.asarray(arima_fit.resid, dtype=float)
        if len(resid) > 50:  # avoid tiny samples
            garch_fit = _fit_garch(resid)
            garch_var = garch_fit.forecast(
                horizon=max_h, reindex=False
            ).variance.values[-1]
            garch_sigma = np.sqrt(garch_var)

    updated_results = arima_fit
    last_appended = train_end

    for t in test_idx:
        if int(t) > last_appended:
            new_obs = close_series[last_appended + 1 : int(t) + 1]
            if len(new_obs) > 0:
                updated_results = updated_results.append(new_obs, refit=False)
                last_appended = int(t)

        mean_forecast = _forecast_arima_mean(updated_results, steps=max_h)
        pred = float(mean_forecast[horizon - 1])
        preds.append(pred)

        if garch_sigma is not None:
            sigmas.append(float(garch_sigma[horizon - 1]))

    y_pred = np.asarray(preds, dtype=float)
    pred_sigma = np.asarray(sigmas, dtype=float) if sigmas else None

    return y_true, y_pred, pred_sigma


def _fold_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, naive_pred: np.ndarray
) -> FoldMetrics:
    rmse_value = _rmse(y_true, y_pred)
    mae_value = float(mean_absolute_error(y_true, y_pred))
    r2_value = float(r2_score(y_true, y_pred))
    rmse_pct_value = _rmse_pct(y_true, rmse_value)

    naive_rmse_value = _rmse(y_true, naive_pred)
    naive_mae_value = float(mean_absolute_error(y_true, naive_pred))
    naive_r2_value = float(r2_score(y_true, naive_pred))
    naive_rmse_pct_value = _rmse_pct(y_true, naive_rmse_value)

    return FoldMetrics(
        rmse=rmse_value,
        mae=mae_value,
        r2=r2_value,
        rmse_pct=rmse_pct_value,
        naive_rmse=naive_rmse_value,
        naive_mae=naive_mae_value,
        naive_r2=naive_r2_value,
        naive_rmse_pct=naive_rmse_pct_value,
    )


def main():
    mlflow = _get_mlflow()

    df_path = Path("data/processed/gold_features.csv")
    if not df_path.exists():
        raise FileNotFoundError(
            "Missing data/processed/gold_features.csv. Run preprocessing first: python src/data/preprocess_gold.py"
        )

    df = pd.read_csv(df_path)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    horizons = [1, 3, 5, 7]
    for h in horizons:
        if f"y_{h}" not in df.columns:
            raise ValueError(
                f"Missing y_{h} in gold_features.csv. Make sure preprocess_gold.py creates multi-horizon targets."
            )

    # Use the same gap concept as your ML models to avoid horizon leakage
    tscv = TimeSeriesSplit(n_splits=5, gap=max(horizons))

    arima_order = (1, 1, 1)

    if mlflow is not None:
        import logging

        logging.getLogger("mlflow").setLevel(logging.ERROR)
        warnings.filterwarnings("ignore", category=UserWarning, module="mlflow")
        mlflow.set_experiment("gold-close-forecast")

    results_rows: list[dict] = []

    for use_garch in (False, True):
        model_name = "ARIMA" if not use_garch else "ARIMA-GARCH"
        print(f"\n{'=' * 70}")
        print(f"{model_name} (order={arima_order})")
        print(f"{'=' * 70}")

        for horizon in horizons:
            fold_metrics: list[FoldMetrics] = []

            for fold, (train_idx, test_idx) in enumerate(tscv.split(df), start=1):
                y_true, y_pred, _sigma = _predict_for_fold(
                    df=df,
                    train_idx=train_idx,
                    test_idx=test_idx,
                    horizon=horizon,
                    arima_order=arima_order,
                    use_garch=use_garch,
                )

                # naive baseline: Close(t+h) ≈ Close(t)
                naive_pred = df["Close"].iloc[test_idx].to_numpy(dtype=float)

                fm = _fold_metrics(y_true, y_pred, naive_pred)
                fold_metrics.append(fm)

                if mlflow is not None:
                    with mlflow.start_run(
                        run_name=f"{model_name}_h{horizon}_fold{fold}"
                    ):
                        mlflow.log_param("model", model_name)
                        mlflow.log_param("horizon", horizon)
                        mlflow.log_param("fold", fold)
                        mlflow.log_param("arima_order", str(arima_order))
                        mlflow.log_param("use_garch", bool(use_garch))
                        mlflow.log_param("n_splits", int(tscv.n_splits))
                        mlflow.log_param("gap", int(max(horizons)))
                        mlflow.log_param("train_size", int(len(train_idx)))
                        mlflow.log_param("test_size", int(len(test_idx)))

                        mlflow.log_metric("rmse", fm.rmse)
                        mlflow.log_metric("mae", fm.mae)
                        mlflow.log_metric("r2", fm.r2)
                        mlflow.log_metric("rmse_pct", fm.rmse_pct)

                        mlflow.log_metric("naive_rmse", fm.naive_rmse)
                        mlflow.log_metric("naive_mae", fm.naive_mae)
                        mlflow.log_metric("naive_r2", fm.naive_r2)
                        mlflow.log_metric("naive_rmse_pct", fm.naive_rmse_pct)

                print(
                    f"h=+{horizon}d fold={fold}: RMSE% {fm.rmse_pct * 100:.3f}% vs Naive {fm.naive_rmse_pct * 100:.3f}%"
                )

            avg = FoldMetrics(
                rmse=float(np.mean([m.rmse for m in fold_metrics])),
                mae=float(np.mean([m.mae for m in fold_metrics])),
                r2=float(np.mean([m.r2 for m in fold_metrics])),
                rmse_pct=float(np.mean([m.rmse_pct for m in fold_metrics])),
                naive_rmse=float(np.mean([m.naive_rmse for m in fold_metrics])),
                naive_mae=float(np.mean([m.naive_mae for m in fold_metrics])),
                naive_r2=float(np.mean([m.naive_r2 for m in fold_metrics])),
                naive_rmse_pct=float(np.mean([m.naive_rmse_pct for m in fold_metrics])),
            )

            print(f"\nAVERAGE (h=+{horizon}d) {model_name}")
            print(f"  RMSE : {avg.rmse:.6f}")
            print(f"  MAE  : {avg.mae:.6f}")
            print(f"  R2   : {avg.r2:.4f}")
            print(f"  RMSE%: {avg.rmse_pct * 100:.3f}%")
            print(f"  Naive RMSE%: {avg.naive_rmse_pct * 100:.3f}%")

            if mlflow is not None:
                with mlflow.start_run(run_name=f"{model_name}_h{horizon}_avg"):
                    mlflow.log_param("model", model_name)
                    mlflow.log_param("horizon", horizon)
                    mlflow.log_param("fold", "avg")
                    mlflow.log_param("arima_order", str(arima_order))
                    mlflow.log_param("use_garch", bool(use_garch))
                    mlflow.log_param("n_splits", int(tscv.n_splits))
                    mlflow.log_param("gap", int(max(horizons)))

                    mlflow.log_metric("rmse", avg.rmse)
                    mlflow.log_metric("mae", avg.mae)
                    mlflow.log_metric("r2", avg.r2)
                    mlflow.log_metric("rmse_pct", avg.rmse_pct)

                    mlflow.log_metric("naive_rmse", avg.naive_rmse)
                    mlflow.log_metric("naive_mae", avg.naive_mae)
                    mlflow.log_metric("naive_r2", avg.naive_r2)
                    mlflow.log_metric("naive_rmse_pct", avg.naive_rmse_pct)

            results_rows.append(
                {
                    "model": model_name,
                    "arima_order": str(arima_order),
                    "horizon": horizon,
                    "rmse": avg.rmse,
                    "mae": avg.mae,
                    "r2": avg.r2,
                    "rmse_pct": avg.rmse_pct,
                    "naive_rmse": avg.naive_rmse,
                    "naive_mae": avg.naive_mae,
                    "naive_r2": avg.naive_r2,
                    "naive_rmse_pct": avg.naive_rmse_pct,
                }
            )

    out_dir = Path("models")
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "arima_results.csv"
    pd.DataFrame(results_rows).to_csv(out_path, index=False)
    print(f"\nSaved ARIMA results to {out_path}")

    if mlflow is not None:
        with mlflow.start_run(run_name="ARIMA_RESULTS"):
            mlflow.log_param("file", "models/arima_results.csv")
            mlflow.log_artifact(str(out_path))


if __name__ == "__main__":
    main()
