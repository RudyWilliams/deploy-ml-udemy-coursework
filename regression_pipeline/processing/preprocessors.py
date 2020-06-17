import numpy as np
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


class NumericalImputer(BaseEstimator, TransformerMixin):
    """used to impute numerical features with NaNs"""

    def __init__(self, variables=None):
        if not isinstance(variables, list):
            self.variables = [variables]
        else:
            self.variables = variables

    def fit(self, X, y=None):
        self.imputer_dict_ = {}
        for feature in self.variables:
            self.imputer_dict_[feature] = X[feature].mode()[0]
        return self

    def transform(self, X):
        X = X.copy()
        for feature in self.variables:
            X[feature].fillna(self.imputer_dict_[feature], inplace=True)
        return X


class TemporalVariableEstimator(BaseEstimator, TransformerMixin):
    """calc btwn year sold and year built"""

    def __init__(self, variables=None, reference_variable=None):
        if not isinstance(variables, list):
            self.variables = [variables]
        else:
            self.variables = variables

        self.reference_variable = reference_variable

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        for feature in self.variables:
            X[feature] = X[self.reference_variable] - X[feature]
        return X


class RareLabelCategoricalEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, tol=0.05, variables=None):
        self.tol = tol
        if not isinstance(variables, list):
            self.variables = [variables]
        else:
            self.variables = variables

    def fit(self, X, y=None):
        self.encoder_dict_ = {}

        for var in self.variables:
            t = pd.Series(X[var].value_counts() / np.float(len(X)))
            # maps features to what is the NONRARE labels
            self.encoder_dict_[var] = list(t[t >= self.tol].index)

        return self

    def transform(self, X):
        X = X.copy()
        for feature in self.variables:
            X[feature] = np.where(
                X[feature].isin(self.encoder_dict_[feature]), X[feature], "Rare"
            )
        return self


class CategoricalEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, variables=None):
        if not isinstance(variables, list):
            self.variables = [variables]
        else:
            self.variables = variables

    def fit(self, X, y):
        # we are using the target to create a numeric value to map each category to
        temp = pd.concat([X, y], axis=1)
        temp.columns = list(X.columns) + ["target"]

        self.encoder_dict_ = {}
        for var in self.variables:
            t = temp.groupby([var])["target"].mean().sort_values(ascending=True).index
            self.encoder_dict_[var] = {k: i for i, k in enumerate(t, 0)}

        return self

    def transform(self, X):
        X = X.copy()
        for feature in self.variables:
            # go from category to numeric value from learned dictionary
            X[feature] = X[feature].map(self.encoder_dict_[feature])

        # check if transformer introduces NaN
        if X[self.variables].isnull().any().any():
            null_counts = X[self.variables].isnull().any()
            vars_ = {
                key: value for (key, value) in null_counts.items() if value is True
            }
            raise ValueError(
                f"Categorical encoder has introduced NaN when "
                f"transforming categorical variables: {vars_.keys()}"
            )
        return X


class LogTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, variables):
        if not isinstance(variables, list):
            self.variables = [variables]
        else:
            self.variables = variables

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        # values must be positive for the logtransform
        if not (X[self.variables] > 0).all().all():
            vars_ = self.variables[(X[self.variables] <= 0).any()]
            raise ValueError(f"Variables contain nonpositive values: {vars_}")

        for feature in self.variables:
            X[feature] = np.log(X[feature])

        return X


class DropUnnecessaryFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, variables_to_drop=None):
        self.variables = variables_to_drop

    def fit(self, X):
        return self

    def transform(self, X):
        X = X.copy()
        X = X.drop(self.variables, axis=1)
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
