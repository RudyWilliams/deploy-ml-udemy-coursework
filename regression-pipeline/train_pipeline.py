import pathlib

PACKAGE_ROOT = pathlib.Path(__file__).resolve().parent
TRAINED_MODEL_DIR = PACKAGE_ROOT / "trained_models"
DATASET_DIR = PACKAGE_ROOT / "data"

TESTING_DATA_FILE = DATASET_DIR / "test.csv"
TRAINING_DATA_FILE = DATASET_DIR / "train.csv"
TARGET = "SalesPrice"

FEATURES = [
    "MSSubClass",
    "MSZoning",
    "Neighborhood",
    "OverallQual",
    "OverallCond",
    "YearRemodAdd",
    "RoofStyle",
    "MasVnrType",
    "BsmtQual",
    "BsmtExposure",
    "HeatingQC",
    "CentralAir",
    "1stFlrSF",
    "GrLivArea",
    "BsmtFullBath",
    "KitchenQual",
    "Fireplaces",
    "FireplaceQu",
    "GarageType",
    "GarageFinish",
    "GarageCars",
    "PavedDrive",
    "LotFrontage",
    "YrSold",
]


def save_pipeline() -> None:
    """will eventually use to persist the pipeline"""
    pass


def run_training() -> None:
    """will eventually train the model"""
    print("Training model...")


if __name__ == "__main__":
    run_training()
