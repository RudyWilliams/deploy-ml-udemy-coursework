import pandas as pd

from regression_pipeline.config import config


def validate_inputs(input_data: pd.DataFrame) -> pd.DataFrame:
    """Check for values that cannot be processed"""
    validated_data = input_data.copy()

    # check for numerical variables with NA not seen during training
    if input_data[config.NUMERICAL_NA_NOT_ALLOWED].isnull().any().any():
        validated_data = validated_data.dropna(
            axis=0, subset=config.NUMERICAL_NA_NOT_ALLOWED
        )

    if input_data[config.CATEGORICAL_NA_NOT_ALLOWED].isnull().any().any():
        validated_data = validated_data.dropna(
            axis=0, subset=config.CATEGORICAL_NA_NOT_ALLOWED
        )

    # check for values <= 0 for log transformation
    is_lte_zero = input_data[config.NUMERICALS_LOG_VARS] <= 0
    if is_lte_zero.any().any():
        vars_with_neg_values = config.NUMERICALS_LOG_VARS[is_lte_zero.any()]
        validated_data = validated_data[validated_data[vars_with_neg_values] > 0]

    return validated_data

