import logging

import numpy as np
import pandas as pd

from regression_pipeline.config import config
from regression_pipeline.processing.data_management import load_pipeline
from regression_pipeline.processing.validation import validate_inputs
from regression_pipeline import __version__ as _version

_logger = logging.getLogger(__name__)
pipeline_file_name = (
    f"{config.PIPELINE_SAVE_FILE}{_version}.pkl"  # include version in pkl save
)
_price_pipe = load_pipeline(file_name=pipeline_file_name)


def make_prediction(*, input_data) -> dict:
    """Make prediction using the saved model pipeline"""

    data = pd.read_json(input_data)
    validated_data = validate_inputs(input_data=data)
    prediction = _price_pipe.predict(validated_data[config.FEATURES])
    output = np.exp(prediction)

    results = {"predictions": output, "version": _version}

    _logger.info(
        f"Making predictions with model version: {_version} "
        f"Inputs: {validated_data} "
        f"Predictions: {results}"
    )

    return results
