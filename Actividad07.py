"""
Seminario de Inteligencia Artificial 2
Actividad 07
"""

import numpy as np
import matplotlib.pyplot as plt

from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB

classifiers = { 'KNN':KNeighborsClassifier(3),
                'SVM':SVC(gamma=2, C=1),
                'GP':GaussianProcessClassifier(1.0 * RBF(1.0)),
                'DT':DecisionTreeClassifier(max_depth=15),
                'MLP':MLPClassifier(alpha=.01, max_iter=1000),
                'Bayes':GaussianNB() }

x, y = make_classification(n_features=2, n_redundant=0, n_informative=2,
                           n_clusters_per_class=1)

rng = np.random.RandomState(2)
x += 1 * rng.uniform(size=x.shape)
linearly_separable = (x, y)

datasets = [make_moons(noise=0.1),
            make_circles(noise=0.1, factor=0.5),
            linearly_separable]

cm = plt.cm.RdBu
cm_bright = ListedColormap(['#FF0000', '#0000FF'])

#Model
modelName = 'Bayes'

figure = plt.figure(figsize=(9,3))
h = 0.02
i = 1

for ds_cnt, ds in enumerate(datasets):
    x, y = ds
    x = StandardScaler().fit_transform(x)
    xTrain, xTest, yTrain, yTest = train_test_split(x,y)
    
    xMin, xMax = x[:, 0].min() - .5, x[:, 0].max() + .5
    yMin, yMax = x[:, 1].min() - .5, x[:, 1].max() + .5
    xx, yy = np.meshgrid(np.arange(xMin, xMax, h),
                         np.arange(yMin, yMax, h))
    
    model = classifiers[modelName]
    ax = plt.subplot(1, 3, i)
    model.fit(xTrain, yTrain)
    scoreTrain = model.score(xTrain, yTrain)
    scoreTest = model.score(xTest, yTest)
    
    if(hasattr(model, "decision_function")):
        zz = model.decision_function(np.c_[xx.ravel(), yy.ravel()])
    else:
        zz = model.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
        
    zz = zz.reshape(xx.shape)
    ax.contourf(xx, yy, zz, cmap=cm, alpha=.8)
    
    ax.scatter(xTrain[:, 0], xTrain[:, 1], c=yTrain, cmap=cm_bright,
               edgecolors='k')
    
    ax.scatter(xTest[:, 0], xTest[:, 1], c=yTest, cmap=cm_bright,
               edgecolors='k', alpha=0.6)
    
    ax.text(xx.max() - .3, yy.min() + .7, '%.2f' % scoreTrain,
            size=15, horizontalalignment='right')
    
    ax.text(xx.max() - .3, yy.min() + .3, '%.2f' % scoreTest,
            size=15, horizontalalignment='right')
    
    i += 1
    
plt.tight_layout()
plt.show()


