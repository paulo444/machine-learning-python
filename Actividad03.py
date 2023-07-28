"""
Seminario de Inteligencia Artificial 2
Actividad 03
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model

data = pd.read_csv('countries.csv')
mexico = data[data.country == 'Mexico']

x = np.asanyarray(mexico[['year']])
y = np.asanyarray(mexico[['lifeExp']])

model = linear_model.LinearRegression()
model.fit(x,y)

yPrediction = model.predict(x)

mexico.plot.scatter(x='year', y='lifeExp')
plt.plot(x, yPrediction, '--r')

#Predicciones
print(model.predict([[2005], [2019], [3019], [-542]]))

