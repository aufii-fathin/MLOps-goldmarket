import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


def _fit_arima(close_train: np.ndarray, order: tuple[int, int, int]):
    try:
        from statsmodels.tsa.arima.model import ARIMA
    except Exception as exc:
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


def main():
    horizons = [1, 3, 5, 7]
    max_h = max(horizons)
    arima_order = (1, 1, 1)

    raw_path = Path("data/raw/gold_prices.csv")
    if not raw_path.exists():
        raise FileNotFoundError(
            "Missing data/raw/gold_prices.csv. Run ingestion first: python src/data/ingestion.py"
        )

    df = pd.read_csv(raw_path)
    if "Date" not in df.columns or "Close" not in df.columns:
        raise ValueError("gold_prices.csv must contain Date and Close columns")

    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").drop_duplicates(subset="Date")

    close = df["Close"].to_numpy(dtype=float)

    arima_fit = _fit_arima(close, arima_order)
    mean_forecast = np.asarray(arima_fit.forecast(steps=max_h), dtype=float)

    print(f"\nARIMA(order={arima_order}) forecast from latest close")
    print(f"Latest date: {df['Date'].iloc[-1].date()} | Latest Close: {close[-1]:.4f}")
    for h in horizons:
        print(f"Predicted Close +{h}d: {float(mean_forecast[h - 1]):.4f}")

    try:
        resid = np.asarray(arima_fit.resid, dtype=float)
        if len(resid) > 50:
            garch_fit = _fit_garch(resid)
            var = garch_fit.forecast(horizon=max_h, reindex=False).variance.values[-1]
            sigma = np.sqrt(var)

            print("\nARIMA-GARCH(1,1) approximate 95% intervals")
            for h in horizons:
                mu = float(mean_forecast[h - 1])
                s = float(sigma[h - 1])
                lo = mu - 1.96 * s
                hi = mu + 1.96 * s
                print(f"+{h}d: mean={mu:.4f} | 95% [{lo:.4f}, {hi:.4f}]")
    except ImportError:
        pass


if __name__ == "__main__":
    main()
