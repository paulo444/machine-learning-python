"""
Seminario de Inteligencia Artificial 2
Actividad 05
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('daily-min-temperatures.csv')

x = np.asanyarray(data[['Temp']])
#plt.plot(x)

p = 1
#plt.scatter(x[p:], x[:-p])
print(np.corrcoef(x[p:].transpose(), x[:-p].transpose()))

#pd.plotting.autocorrelation_plot(data.Temp)

dataSet = pd.DataFrame(data.Temp)

p = 3
for i in range(1, p+1):
    dataSet = pd.concat([dataSet, data.Temp.shift(-i)], axis=1)

dataSet = dataSet[:-p]

print(dataSet.head())

x = np.asanyarray(dataSet.iloc[:, 1:])
y = np.asanyarray(dataSet.iloc[:, 0])

from sklearn.model_selection import train_test_split
xTrain, xTest, yTrain, yTest = train_test_split(x,y)

from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(xTrain, yTrain)

print("Train: ", model.score(xTrain, yTrain))
print("Test: ", model.score(xTest, yTest))

print(model.predict([[20,18,17]]))
print(model.predict([[15,30,19]]))
print(model.predict([[18,24,16]]))

