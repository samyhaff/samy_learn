from copy import deepcopy
import numpy as np


def gradient_descent(gradient, start, alpha=1e-03, iter_max=100, tolerance=1e-07):
    """Gradient Descent algorithm"""
    x = start
    for _ in range(iter_max):
        diff = alpha * gradient(x)
        if np.max(abs(diff)) < tolerance:
            break
        x -= diff
    return x


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


class GDRegressor:
    """Gradient Dezcent Linear Regression"""

    def __init__(self):
        self.w = None

    def gradient(self, x, X, y):
        X = deepcopy(X)
        X = np.c_[X, np.ones(X.shape[0])]
        return 2 * X.T @ (X @ x - y)

    def fit(self, X, y):
        """fit the model using OLS"""
        start = np.zeros(X.shape[1] + 1)
        self.w = gradient_descent(lambda x: self.gradient(x, X, y), start)

    def predict(self, X):
        """make a prediction using a fitted model"""
        if self.w is None:
            raise Exception("The model is not fitted")
        X = deepcopy(X)
        X = np.c_[X, np.ones(X.shape[0])]
        return X @ self.w
