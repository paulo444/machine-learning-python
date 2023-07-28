"""
Seminario de Inteligencia Artificial 2
Actividad 06
"""

import numpy as np
import numpy.random as rnd
import matplotlib.pyplot as plt

np.random.seed(42)
m = 300
r = 0.5
noise = r * np.random.randn(m, 1)
x = 6 * np.random.rand(m, 1) - 3
y = 0.5 * x**2 + x + 2 + noise

from sklearn.model_selection import train_test_split
xTrain, xTest, yTrain, yTest = train_test_split(x, y)
"""
plt.plot(xTrain, yTrain, "b.")
plt.plot(xTest, yTest, "r.")
plt.xlabel("$x$", fontsize=18)
plt.ylabel("$y$", fontsize=18)
plt.axis([-3, 3, 0, 10])
plt.show()
"""

#DecisionTree
"""
from sklearn.tree import DecisionTreeRegressor
model = DecisionTreeRegressor(max_depth=10, min_samples_leaf=0.05)
model.fit(xTrain, yTrain)

print("Train: ", model.score(xTrain, yTrain))
print("Test: ", model.score(xTest, yTest))

xNew = np.linspace(-3, 3, 50).reshape(-1, 1)
yPred = model.predict(xNew)

plt.plot(xNew, yPred, "k-", linewidth=3)
plt.plot(xTrain, yTrain, "b.")
plt.plot(xTest, yTest, "r.")
plt.xlabel("$x$", fontsize=18)
plt.ylabel("$y$", fontsize=18)
plt.axis([-3, 3, 0, 10])
plt.show()
"""

"""
#K-Nearest Neighbor
from sklearn.neighbors import KNeighborsRegressor
model = KNeighborsRegressor(n_neighbors=5, weights="uniform")
model.fit(xTrain, yTrain)

print("Train: ", model.score(xTrain, yTrain))
print("Test: ", model.score(xTest, yTest))

xNew = np.linspace(-3, 3, 50).reshape(-1, 1)
yPred = model.predict(xNew)

plt.plot(xNew, yPred, "k-", linewidth=3)
plt.plot(xTrain, yTrain, "b.")
plt.plot(xTest, yTest, "r.")
plt.xlabel("$x$", fontsize=18)
plt.ylabel("$y$", fontsize=18)
plt.axis([-3, 3, 0, 10])
plt.show()
"""

"""
#SVR
from sklearn.svm import SVR
model = SVR(gamma="scale", C=10, epsilon=0.1, kernel="rbf")
model.fit(xTrain, yTrain.ravel())

print("Train: ", model.score(xTrain, yTrain.ravel()))
print("Test: ", model.score(xTest, yTest).ravel())

xNew = np.linspace(-3, 3, 50).reshape(-1, 1)
yPred = model.predict(xNew)

plt.plot(xNew, yPred, "k-", linewidth=3)
plt.plot(xTrain, yTrain, "b.")
plt.plot(xTest, yTest, "r.")
plt.xlabel("$x$", fontsize=18)
plt.ylabel("$y$", fontsize=18)
plt.axis([-3, 3, 0, 10])
plt.show()
"""

"""
#Kernel Ridge
from sklearn.kernel_ridge import KernelRidge
model = KernelRidge(alpha=0.1, kernel="rbf")
model.fit(xTrain, yTrain.ravel())

print("Train: ", model.score(xTrain, yTrain.ravel()))
print("Test: ", model.score(xTest, yTest).ravel())

xNew = np.linspace(-3, 3, 50).reshape(-1, 1)
yPred = model.predict(xNew)

plt.plot(xNew, yPred, "k-", linewidth=3)
plt.plot(xTrain, yTrain, "b.")
plt.plot(xTest, yTest, "r.")
plt.xlabel("$x$", fontsize=18)
plt.ylabel("$y$", fontsize=18)
plt.axis([-3, 3, 0, 10])
plt.show()
"""

#MLP
from sklearn.neural_network import MLPRegressor
model = MLPRegressor(hidden_layer_sizes=(100,20), solver="adam",
                              activation="relu", batch_size=10)
model.fit(xTrain, yTrain.ravel())

print("Train: ", model.score(xTrain, yTrain.ravel()))
print("Test: ", model.score(xTest, yTest).ravel())

xNew = np.linspace(-3, 3, 50).reshape(-1, 1)
yPred = model.predict(xNew)

plt.plot(xNew, yPred, "k-", linewidth=3)
plt.plot(xTrain, yTrain, "b.")
plt.plot(xTest, yTest, "r.")
plt.xlabel("$x$", fontsize=18)
plt.ylabel("$y$", fontsize=18)
plt.axis([-3, 3, 0, 10])
plt.show()