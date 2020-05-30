import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class CategoricalImputer(BaseEstimator, TransformerMixin):
    """
     Categorical imputer is just filling the NaNs with "Missing" bc missing
     information is information.
    """

    def __init__(self, variables=None) -> None:
        if not isinstance(variables, list):
            # allows a string to be passed in the case of a single var
            self.variables = [variables]
        else:
            self.variables = variables

    def fit(
        self, X: pd.DataFrame, y: pd.Series = None
    ) -> "CategoricalImputer":  # bc it returns self
        # no 'learning' is done, just need for sklearn pipeline accommodation
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        # not returning self this time but a dataframe instead
        X = X.copy()
        # I'm doing away with the for loop here
        X[self.variables] = X[self.variables].fillna("Missing")
        return X


if __name__ == "__main__":
    import numpy as np

    df = pd.DataFrame(
        {
            "c1": [1, 2, 3, 4, 5],
            "c2": ["a", "b", "c", np.nan, "e"],
            "c3": [1, 2, np.nan, 4, 5],
            "c4": [np.nan, "b", "c", np.nan, "e"],
        }
    )
    ci = CategoricalImputer(variables=["c2", "c4"])
    print(ci.transform(df))
