import numpy as np
import pandas as pd

from regression_pipeline.config import config
from regression_pipeline.processing.data_management import load_pipeline
from regression_pipeline.processing.validation import validate_inputs


pipeline_file_name = "regression_pipeline.pkl"
_price_pipe = load_pipeline(file_name=pipeline_file_name)


def make_prediction(*, input_data) -> dict:
    """Make prediction using the saved model pipeline"""

    data = pd.read_json(input_data)
    validated_data = validate_inputs(input_data=data)
    prediction = _price_pipe.predict(validated_data[config.FEATURES])
    output = np.exp(prediction)
    response = {"predictions": output}

    return response
