from copy import deepcopy
import numpy as np


def sigmoid(x):
    """Sigmoid function"""
    return 1 / (1 + np.exp(-x))


def mse(y_hat, y):
    """Mean Square Error"""
    return np.sum(((y - y_hat) ** 2)) / y.shape[0]


def gradient_descent(
    gradient, start, learning_rate=1e-03, iter_max=100, tolerance=1e-07
):
    """Gradient Descent algorithm"""
    x = start
    for _ in range(iter_max):
        diff = learning_rate * gradient(x)
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
    """Gradient Dezcent Linear Regression model"""

    def __init__(self):
        self.w = None

    def gradient(self, x, X, y):
        """compute the gradient"""
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


class Ridge:
    """Ridge Regression model"""

    def __init__(self, alpha=1.0):
        self.w = None
        self.alpha = alpha

    def gradient(self, x, X, y):
        """compute the gradient"""
        X = deepcopy(X)
        X = np.c_[X, np.ones(X.shape[0])]
        return 2 * (X.T @ (X @ x - y) + self.alpha * x)

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


class LogisticRegression:
    """Logistic Regression model"""

    def __init__(self):
        self.w = None

    def gradient(self, x, X, y):
        """compute the gradient of the log likelihood"""
        X = deepcopy(X)
        X = np.c_[X, np.ones(X.shape[0])]
        return X.T @ (y - sigmoid(X @ x))

    def fit(self, X, y):
        """fit the model using gradient ascent"""
        start = np.zeros(X.shape[1] + 1)
        self.w = gradient_descent(lambda x: -self.gradient(x, X, y), start)

    def predict_proba(self, X):
        """returns predicted probabilities"""
        if self.w is None:
            raise Exception("The model is not fitted")
        X = deepcopy(X)
        X = np.c_[X, np.ones(X.shape[0])]
        return sigmoid(X @ self.w)

    def predict(self, X, threshold=0.5):
        """return prediction"""
        if self.w is None:
            raise Exception("The model is not fitted")
        return (self.predict_proba(X) >= threshold).astype("int")


class KMeans:
    """K-means clustering model"""

    def __init__(self, n_clusters):
        self.n_clusters = n_clusters
        self.labels = None
        self.centroids = None

    @staticmethod
    def cluster(X, centroids):
        """cluster according to centroids"""
        vectors = [[x - centroid for centroid in centroids] for x in X]
        distances = [[np.linalg.norm(v) for v in vector] for vector in vectors]
        labels = np.argmin(distances, axis=1)
        return labels

    @staticmethod
    def update_centroids(X, labels):
        """update centroids according to current labels"""
        centroids = np.array([np.mean(X[labels == label], axis=0) for label in labels])
        return centroids

    def fit(self, X):
        """fit the model"""
        n_samples, _ = X.shape

        # generate random centroids
        rng = np.random.default_rng()
        idx = rng.choice(n_samples, size=self.n_clusters, replace=False)
        centroids = X[idx, :]

        labels_old = None
        labels = self.cluster(X, centroids)
        print(labels)

        while np.all(labels != labels_old):
            labels_old = deepcopy(labels)
            centroids = self.update_centroids(X, labels)
            labels = self.cluster(X, centroids)

        self.labels = labels
        self.centroids = centroids

    def predict(self, X):
        """predict labels"""
        if self.labels is None:
            raise Exception("The model is not fitted")
        return self.cluster(X, self.centroids)
