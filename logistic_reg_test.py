import numpy as np
from sklearn import datasets
from samy_learn import LogisticRegression
import matplotlib.pyplot as plt

X, y = datasets.make_classification(n_samples=100, n_features=1, n_redundant=0, n_informative=1, n_clusters_per_class=1, random_state=42)
model = LogisticRegression()
model.fit(X, y)
y_pred = model.predict(X)

plt.scatter(X, y, label='Data')
plt.scatter(X, y_pred, label="Prediction")
plt.legend()
plt.show()
