from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.ingestion import main as ingest
from src.data.preprocess_gold import main as preprocess_gold
from src.data.preprocess_oil import main as preprocess_oil
from src.data.preprocess_fred import main as preprocess_fred
from src.data.preprocess_merge import main as merge_datasets

def run_data_pipeline():

    print("Running data ingestion...")
    ingest()

    print("Preprocessing gold data...")
    preprocess_gold()

    print("Preprocessing oil data...")
    preprocess_oil()

    print("Preprocessing macro data...")
    preprocess_fred()

    print("Merging datasets...")
    merge_datasets()

    print("Data pipeline finished.")


if __name__ == "__main__":
    run_data_pipeline()