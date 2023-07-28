"""
Seminario de Inteligencia Artificial 2
Actividad 09
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline

data = pd.read_csv("mnist_784.csv")
nSamples = 30000

x = np.asanyarray(data.drop(columns=["class"]))[:nSamples, :]
y = np.asanyarray(data[["class"]])[:nSamples].ravel()

xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size=0.1)

model = Pipeline([
    ("scaler", StandardScaler()),
    ("pca", PCA(n_components=50)),
    ("svm", svm.SVC(gamma=0.0001))
    ])

model.fit(xTrain, yTrain)
print("Train: ", model.score(xTrain, yTrain))
print("Test: ", model.score(xTest, yTest))

yPred = model.predict(xTest)

print("Classification Report: \n", metrics.classification_report(yTest, yPred))
print("Confusion Matrix: \n", metrics.confusion_matrix(yTest, yPred))

sample = np.random.randint(xTest.shape[0])
plt.imshow(xTest[sample].reshape((28,28)), cmap=plt.cm.gray)
plt.title("Prediction: %i" % yPred[sample])
plt.show()

