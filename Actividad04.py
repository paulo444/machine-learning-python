"""
Seminario de Inteligencia Artificial 2
Actividad 04
"""

import numpy as np
import matplotlib.pyplot as plt

#Crear datos y graficarlos
np.random.seed(42)

m = 100
x = 6 * np.random.rand(m, 1) - 3
y = .5 * x**2 + x + 2 + np.random.rand(m, 1)

plt.plot(x, y, "b.")
plt.xlabel("$x_1$", fontsize=18)
plt.ylabel("$y$", rotation=0, fontsize=18)
plt.axis([-3, 3, 0, 10])
plt.show()

#Obtener datos polinomiales
from sklearn.preprocessing import PolynomialFeatures
polyFeatures = PolynomialFeatures(degree=2, include_bias=False)
xPoly = polyFeatures.fit_transform(x)
print(xPoly[0])

#Datos de la regresión lineal
from sklearn.linear_model import LinearRegression
linReg = LinearRegression()
linReg.fit(xPoly, y)
print(linReg.intercept_, linReg.coef_)

#Graficar prediccion del polinomio
xNew = np.linspace(-3, 3, 100).reshape(100, 1)
xNewPoly = polyFeatures.transform(xNew)
yNew = linReg.predict(xNewPoly)
plt.plot(x, y, "b.")
plt.plot(xNew, yNew, "r-", linewidth=2, label="Predictions")
plt.xlabel("$x_1$", fontsize=18)
plt.ylabel("$y$", rotation=0, fontsize=18)
plt.legend(loc="upper left", fontsize=14)
plt.axis([-3, 3, 0 , 10])
plt.show()

#Organizar en pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

model = Pipeline([
    ("poly", PolynomialFeatures(degree=1, include_bias=False)),
    ("scaler", StandardScaler()),
    ("reg", LinearRegression())])

model.fit(x, y)

yNew = model.predict(xNew)

plt.plot(xNew, yNew, "r-")

plt.plot(x, y, "b.", linewidth=3)
plt.xlabel("$x_1$")
plt.ylabel("$y$")
plt.axis([-3, 3, 0 , 10])
plt.show()

#Graficar multiples polinomios
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

for style, width, degree in (("g-", 1, 300), ("k--", 2, 2), ("r--", 2, 1)):
    polyBigFeatures = PolynomialFeatures(degree=degree, include_bias=False)
    stdScaler = StandardScaler()
    linReg = LinearRegression()
    polynomialRegression = Pipeline([
        ("poly features", polyBigFeatures),
        ("std_scaler", stdScaler),
        ("lin_reg", linReg)
        ])

    polynomialRegression.fit(x, y)
    yNewBig = polynomialRegression.predict(xNew)
    plt.plot(xNew, yNewBig, style, label=str(degree), linewidth=width)
    
plt.plot(x, y, "b.", linewidth=3)
plt.legend(loc="upper left")
plt.xlabel("$x_1$")
plt.ylabel("$y$")
plt.axis([-3, 3, 0 , 10])
plt.show()

#Resultados evaluados
from sklearn.model_selection import train_test_split

xTrain, xTest, yTrain, yTest = train_test_split(x, y)
model = Pipeline([
    ("poly", PolynomialFeatures(degree=2, include_bias=False)),
    ("scaler", StandardScaler()),
    ("reg", LinearRegression())])

model.fit(xTrain, yTrain)

print("Train: ", model.score(xTrain, yTrain))
print("Test: ", model.score(xTest, yTest))

yNew = model.predict(xNew)
plt.plot(xNew, yNew, "k-")
plt.plot(xTrain, yTrain, "b.")
plt.plot(xTest, yTest, "r.")
plt.xlabel("$x_1$")
plt.ylabel("$y$")
plt.axis([-3, 3, 0 , 10])
plt.show()

#Regularización de los datos con Ridge
from sklearn.linear_model import Ridge

np.random.seed(42)
m = 20
x = 3 * np.random.rand(m, 1)
y = 1 + 0.5 * x + np.random.randn(m, 1) / 1.5
xNew = np.linspace(0, 3, 100).reshape(100, 1)

alpha = 0.01

model = Pipeline([
        ("poly features", PolynomialFeatures(degree=100, include_bias=False)),
        ("std_scaler", StandardScaler()),
        ("regular_reg", Ridge(alpha))
        ])

model.fit(x, y)
yNewRegul = model.predict(xNew)
plt.plot(xNew, yNewRegul, "--r", linewidth=3, label=r"$\alpha = {}$".format(alpha))
plt.plot(x, y, "ob")
plt.legend()
plt.xlabel("$x_1$")
plt.ylabel("$y$")
plt.axis([0, 3, 0 , 4])
plt.show()

#Regularización de los datos con Lasso
from sklearn.linear_model import Lasso

alpha = 0.01

model = Pipeline([
        ("poly features", PolynomialFeatures(degree=100, include_bias=False)),
        ("std_scaler", StandardScaler()),
        ("regular_reg", Lasso(alpha))
        ])

model.fit(x, y)
yNewRegul = model.predict(xNew)
plt.plot(xNew, yNewRegul, "--r", linewidth=3, label=r"$\alpha = {}$".format(alpha))
plt.plot(x, y, "ob")
plt.legend()
plt.xlabel("$x_1$")
plt.ylabel("$y$")
plt.axis([0, 3, 0 , 4])
plt.show()

#Regularización de los datos con ElasticNet
from sklearn.linear_model import ElasticNet

model = Pipeline([
        ("poly features", PolynomialFeatures(degree=10, include_bias=False)),
        ("std_scaler", StandardScaler()),
        ("regular_reg", ElasticNet(alpha=0.001, l1_ratio=0.001, random_state=42))
        ])

model.fit(x, y)
yNewRegul = model.predict(xNew)
plt.plot(xNew, yNewRegul, "--r", linewidth=3, label=r"$\alpha = {}$".format(alpha))
plt.plot(x, y, "ob")
plt.legend()
plt.xlabel("$x_1$")
plt.ylabel("$y$")
plt.axis([0, 3, 0 , 4])
plt.show()