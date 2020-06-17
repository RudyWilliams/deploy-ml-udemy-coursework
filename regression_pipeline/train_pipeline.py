import numpy as np
from sklearn.model_selection import train_test_split

from regression_pipeline import pipeline
from regression_pipeline.config import config
from regression_pipeline.processing.data_management import load_dataset, save_pipeline


def run_training() -> None:
    """Train the model"""
    data = load_dataset(file_name=config.TRAINING_DATA_FILE)

    X_train, X_test, y_train, y_test = train_test_split(
        data[config.FEATURES], data[config.TARGET], test_size=0.1, random_state=0
    )

    # print(type(X_train))
    # print(type(y_train))
    y_train = np.log(y_train)
    # print(type(y_train))
    pipeline.price_pipe.fit(X_train[config.FEATURES], y_train)
    save_pipeline(pipeline_to_persist=pipeline.price_pipe)


if __name__ == "__main__":
    run_training()
