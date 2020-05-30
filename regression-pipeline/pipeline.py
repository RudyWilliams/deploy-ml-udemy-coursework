from sklearn.pipeline import Pipeline

import preprocessors as pp


CATEGORICAL_VARS = [
    "MSZoning",
    "Neighborhood",
    "RoofStyle",
    "MasVnrType",
    "BsmtQual",
    "BsmtExposure",
    "HeatingQC",
    "CentralAir",
    "KitchenQual",
    "FireplaceQu",
    "GarageType",
    "GarageFinish",
    "PavedDrive",
]

PIPELINE_NAME = "lasso_regression"

price_pipeline = Pipeline(
    # pass a list of tuples with the first element being the name to call the step
    # (name, transform) - the pipeline calls fit_transform on each in order
    [("categorical_imputer", pp.CategoricalImputer(variables=CATEGORICAL_VARS))]
)
