import numpy as np
from sklearn import datasets
from samy_learn import LinearRegression, GDRegressor, SGDRegressor
import matplotlib.pyplot as plt

X, y = datasets.make_regression(n_samples=100, n_features=1, noise=20, random_state=42)
model = SGDRegressor()
model.fit(X, y)
y_pred = model.predict(X)

plt.scatter(X, y)
plt.plot(X, y_pred)
plt.show()
