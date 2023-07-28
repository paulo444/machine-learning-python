"""
Seminario de Inteligencia Artificial 2
Actividad 08
"""

import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler

classifiers = { 'KNN':KNeighborsClassifier(3),
                'SVM':SVC(gamma=0.0001),
                'GP':GaussianProcessClassifier(1.0 * RBF(1.0)),
                'DT':DecisionTreeClassifier(max_depth=5),
                'MLP':MLPClassifier(alpha=.01, max_iter=1000),
                'Bayes':GaussianNB() }

data = pd.read_csv('diabetes.csv')

target, information = data['Outcome'].values, data.values

information = np.delete(information, np.s_[-1], axis=1)

information = StandardScaler().fit_transform(information)

xTrain, xTest, yTrain, yTest = train_test_split(information, target)

model = classifiers['Bayes']

model.fit(xTrain, yTrain)

print('Train: ', model.score(xTrain, yTrain))
print('Test: ', model.score(xTest, yTest))

yPred = model.predict(xTest)

print('Classification report: \n', metrics.classification_report(yTest, yPred))

print('Confusion matrix: \n', metrics.confusion_matrix(yTest, yPred))


