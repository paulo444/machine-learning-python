"""
Seminario de Inteligencia Artificial 2
Actividad 07-1
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split

digits = datasets.load_digits()
target, images = digits['target'], digits['images']

nSamples = digits['target'].shape[0]

sample = np.random.randint(nSamples)
plt.imshow(images[sample])
plt.title('Target: %i' % target[sample])

x = images.reshape((nSamples, -1))

xTrain, xTest, yTrain, yTest = train_test_split(x, target)

model = svm.SVC(gamma=0.0001)

model.fit(xTrain, yTrain)

print('Train: ', model.score(xTrain, yTrain))
print('Test: ', model.score(xTest, yTest))

yPred = model.predict(xTest)

print('Classification report: \n', metrics.classification_report(yTest, yPred))

print('Confusion matrix: \n', metrics.confusion_matrix(yTest, yPred))

sample = np.random.randint(xTest.shape[0])
plt.imshow(xTest[sample].reshape((8,8)))
plt.title('Prediction: %i' % yPred[sample])



