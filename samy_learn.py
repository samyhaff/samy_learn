from copy import deepcopy
import numpy as np

class LinearRegression:
    """Linear Regression model"""

    def __init__(self):
        self.w = None

    def fit(self, X, y):
        """fit the model using OLS"""
        X = deepcopy(X)
        X = np.c_[X, np.ones(X.shape[0])]
        self.w = np.linalg.inv(X.T @ X) @ X.T @ y

    def predict(self, X):
        """make a prediction using a fitted model"""
        if self.w is None:
            raise Exception("The model is not fitted")
        X = deepcopy(X)
        X = np.c_[X, np.ones(X.shape[0])]
        return X @ self.w
