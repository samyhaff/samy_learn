import numpy as np
from sklearn import datasets
from samy_learn import KMeans
import matplotlib.pyplot as plt

X, y = datasets.make_blobs(n_samples=100, centers=3, n_features=2, random_state=42)
model = KMeans(n_clusters=3)
model.fit(X)
labels = model.labels

plt.subplot(2, 1, 1)
plt.scatter(X[:, 0], X[:, 1], c=y)
plt.title('Original')
plt.subplot(2, 1, 2)
plt.scatter(X[:, 0], X[:, 1], c=labels)
plt.title('Kmeans')

plt.show()
