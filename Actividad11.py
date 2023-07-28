"""
Seminario de Inteligencia Artificial 2
Actividad 11
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

data = pd.read_csv("mnist_784.csv")
nSamples = 3000

x = np.asanyarray(data.drop(columns=["class"]))[:nSamples,:]
y = np.asanyarray(data[["class"]])[:nSamples].ravel()

model = TSNE(n_components=2, n_iter=2000)

x2d = model.fit_transform(x)
x2d.shape

plt.scatter(x2d[:,0], x2d[:,1], c=y, cmap=plt.cm.tab10)

