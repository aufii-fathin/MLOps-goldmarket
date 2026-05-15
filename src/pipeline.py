import argparse
import runpy
import sys

from data.data_pipeline import run_data_pipeline
from models.train import main as train
from models.train_arima import main as train_arima
from models.predict import main as predict
from models.predict_arima import main as predict_arima
from registry.transition_stage import main as transition


def _run_module(module_name: str) -> None:
    runpy.run_module(module_name, run_name="__main__")


def _run_pytest() -> None:
    import pytest

    exit_code = pytest.main(["tests"])
    if exit_code != 0:
        sys.exit(exit_code)


def run_pipeline(
    run_data: bool = True,
    run_train: bool = True,
    run_train_arima: bool = False,
    run_evaluate: bool = False,
    run_tests: bool = False,
    run_transition: bool = False,
    run_predict: bool = False,
    run_predict_arima: bool = False,
) -> None:
    if run_data:
        print("Running data pipeline...")
        run_data_pipeline()

    if run_train:
        print("Training ML models...")
        train()

    if run_train_arima:
        print("Training ARIMA models...")
        train_arima()

    if run_evaluate:
        print("Evaluating models...")
        _run_module("models.evaluate")

    if run_tests:
        print("Running pytest...")
        _run_pytest()

    if run_transition:
        print("Registering models...")
        transition()

    if run_predict:
        print("Running ML predictions...")
        predict()

    if run_predict_arima:
        print("Running ARIMA predictions...")
        predict_arima()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MLOps pipeline steps.")
    parser.add_argument("--all", action="store_true", help="Run all steps.")
    parser.add_argument("--data", action="store_true", help="Run data pipeline.")
    parser.add_argument("--train", action="store_true", help="Train ML models.")
    parser.add_argument(
        "--train-arima", action="store_true", help="Train ARIMA models."
    )
    parser.add_argument("--evaluate", action="store_true", help="Run evaluation.")
    parser.add_argument("--pytest", action="store_true", help="Run tests.")
    parser.add_argument("--transition", action="store_true", help="Register models.")
    parser.add_argument("--predict", action="store_true", help="Run ML prediction.")
    parser.add_argument(
        "--predict-arima", action="store_true", help="Run ARIMA prediction."
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    any_flags = any(
        [
            args.all,
            args.data,
            args.train,
            args.train_arima,
            args.evaluate,
            args.pytest,
            args.transition,
            args.predict,
            args.predict_arima,
        ]
    )

    if args.all:
        run_pipeline(
            run_data=True,
            run_train=True,
            run_train_arima=True,
            run_evaluate=True,
            run_tests=True,
            run_transition=True,
            run_predict=True,
            run_predict_arima=True,
        )
    elif not any_flags:
        run_pipeline()
    else:
        run_pipeline(
            run_data=args.data,
            run_train=args.train,
            run_train_arima=args.train_arima,
            run_evaluate=args.evaluate,
            run_tests=args.pytest,
            run_transition=args.transition,
            run_predict=args.predict,
            run_predict_arima=args.predict_arima,
        )
