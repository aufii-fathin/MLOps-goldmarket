from data.data_pipeline import run_data_pipeline
from models.train import main as train
from models.predict import main as predict


def run_pipeline():

    print("Running data pipeline...")
    run_data_pipeline()

    print("Training model...")
    train()


if __name__ == "__main__":
    run_pipeline()