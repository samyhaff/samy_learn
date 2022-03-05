import matplotlib.pyplot as plt
import numpy as np
from samy_learn import LinearRegression, GDRegressor

X = np.array([[0], [1], [2], [3]])
y = np.array([0, 1.5, 1.9, 3])

model = GDRegressor()
model.fit(X, y)
y_pred = model.predict(X)

plt.plot(X, y, 'o', label='Data')
plt.plot(X, y_pred, label='Linear Regression')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()
