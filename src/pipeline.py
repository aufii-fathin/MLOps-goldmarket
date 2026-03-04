from data.ingestion import main as ingestion
from data.preprocess import main as preprocess
from models.train import main as train
from models.predict import main as predict

def run_pipeline():

    print("Running ingestion...")
    ingestion()

    print("Running preprocessing...")
    preprocess()

    print("Training model...")
    train()

    print("Running prediction...")
    predict()

if __name__ == "__main__":
    run_pipeline()