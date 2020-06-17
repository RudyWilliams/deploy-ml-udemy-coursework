import pandas as pd
import joblib
from sklearn.pipeline import Pipeline

from regression_pipeline.config import config


def load_dataset(*, file_name: str) -> pd.DataFrame:
    _data = pd.read_csv(config.DATASET_DIR / file_name)
    return _data


def save_pipeline(*, pipeline_to_persist) -> None:
    save_file_name = "regression_pipeline.pkl"
    save_path = config.TRAINED_MODEL_DIR / save_file_name
    joblib.dump(pipeline_to_persist, save_path)
    print("pipline saved")


def load_pipeline(*, file_name: str) -> Pipeline:
    file_path = config.TRAINED_MODEL_DIR / file_name
    saved_pipeline = joblib.load(filename=file_path)
    return saved_pipeline
