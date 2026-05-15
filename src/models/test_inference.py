from pathlib import Path
import joblib


def main() -> None:
    model_dir = Path("models")
    horizons = [1, 3, 5, 7]

    any_found = False
    for h in horizons:
        model_path = model_dir / f"best_model_h{h}.pkl"
        scaler_path = model_dir / f"scaler_h{h}.pkl"

        if h == 1 and (not model_path.exists() or not scaler_path.exists()):
            model_path = model_dir / "best_model.pkl"
            scaler_path = model_dir / "scaler.pkl"

        if not model_path.exists() or not scaler_path.exists():
            print(f"Missing artifacts for horizon=+{h}d")
            continue

        any_found = True
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        print(
            f"Loaded horizon=+{h}d model: {type(model).__name__}, scaler: {type(scaler).__name__}"
        )

    if not any_found:
        raise FileNotFoundError("No trained model artifacts found. Run training first.")


if __name__ == "__main__":
    main()
