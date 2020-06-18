import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


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
