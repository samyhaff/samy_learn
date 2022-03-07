import numpy as np
from sklearn import datasets
from samy_learn import KNeighborsClassifier
import matplotlib.pyplot as plt

X, y = datasets.make_blobs(n_samples=100, centers=2, n_features=2, random_state=42)

model = KNeighborsClassifier(n_neighbors=3)
model.fit(X, y)

X_new = np.array([[0, 6], [1, 2]])
y_new = model.predict(X_new)

plt.subplot(2, 1, 1)
plt.scatter(X[:, 0], X[:, 1], c=y)
plt.scatter(X_new[:, 0], X_new[:, 1], c=y_new)

plt.show()
