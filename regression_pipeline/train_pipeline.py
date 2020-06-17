import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import joblib

from regression_model import pipeline
from regression_model.config import config


# asterisk means that pipeline to persist is a keyword only argument
def save_pipeline(*, pipeline_to_persist) -> None:
    """Persist the pipeline"""
    save_file_name = "regression_model.pkl"
    save_path = config.TRAINED_MODEL_DIR / save_file_name
    joblib.dump(pipeline_to_persist, save_path)
    print("Pipeline saved")


def run_training() -> None:
    """Train the model"""
    data = pd.read_csv(config.DATASET_DIR / config.TRAINING_DATA_FILE)

    X_train, X_test, y_train, y_test = train_test_split(
        data[config.FEATURES], data[config.TARGET], test_size=0.1, random_state=0
    )

    y_train = np.log(y_train)

    pipeline.price_pipe.fit(X_train[config.FEATURES], y_train)
    save_pipeline(pipeline_to_persist=pipeline.price_pipe)


if __name__ == "__main__":
    run_training()