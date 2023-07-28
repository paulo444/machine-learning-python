"""
Seminario de Inteligencia Artificial 2
Actividad 10
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn import cluster, datasets, mixture, metrics
from sklearn.preprocessing import StandardScaler

np.random.seed(0)
nSamples = 1500
X = 6*[None]

xTemp, _ = datasets.make_circles(n_samples=nSamples, factor=.5, noise=.05)
X[0] = StandardScaler().fit_transform(xTemp)

xTemp, _ = datasets.make_moons(n_samples=nSamples, noise=.05)
X[1] = StandardScaler().fit_transform(xTemp)

xTemp, _ = datasets.make_blobs(n_samples=nSamples, random_state=8)
X[2] = StandardScaler().fit_transform(xTemp)

xTemp = np.random.rand(nSamples, 2)
X[3] = StandardScaler().fit_transform(xTemp)

xTemp, _ = datasets.make_blobs(n_samples=nSamples, random_state=170)
xTemp = np.dot(xTemp, [[0.6, -0.6], [-0.4, 0.8]])
X[4] = StandardScaler().fit_transform(xTemp)

xTemp, _ = datasets.make_blobs(n_samples=nSamples, cluster_std=[1.0, 2.5, 0.5], random_state=142)
X[5] = StandardScaler().fit_transform(xTemp)

classes = [2,2,3,3,3,3]

#plt.figure(figsize=(27,9))
#for i in range(6):
#    ax = plt.subplot(2, 3, i+1)
#    ax.scatter(X[i][:,0], X[i][:,1])
   
'''
#Kmeans
y = []
for c, x in zip(classes, X):
    model = cluster.KMeans(n_clusters=c)
    model.fit(x)
    if hasattr(model, 'labels_'):
        y.append(model.labels_.astype(np.int))
    else:
        y.append(model.predict(x))

fig = plt.figure(figsize=(27,9))
fig.suptitle('Kmeans', fontsize=48)
for i in range(6):
    ax = plt.subplot(2, 3, i+1)
    ax.scatter(X[i][:,0], X[i][:,1], c=y[i])
'''

'''
#SpectralClustering
y = []
for c, x in zip(classes, X):
    model = cluster.SpectralClustering(n_clusters=c)
    model.fit(x)
    if hasattr(model, 'labels_'):
        y.append(model.labels_.astype(np.int))
    else:
        y.append(model.predict(x))

fig = plt.figure(figsize=(27,9))
fig.suptitle('SpectralClustering', fontsize=48)
for i in range(6):
    ax = plt.subplot(2, 3, i+1)
    ax.scatter(X[i][:,0], X[i][:,1], c=y[i])
'''    

'''
#SpectralClustering
y = []
for c, x in zip(classes, X):
    model = cluster.SpectralClustering(n_clusters=c, affinity="nearest_neighbors")
    model.fit(x)
    if hasattr(model, 'labels_'):
        y.append(model.labels_.astype(np.int))
    else:
        y.append(model.predict(x))

fig = plt.figure(figsize=(27,9))
fig.suptitle('SpectralClustering', fontsize=48)
for i in range(6):
    ax = plt.subplot(2, 3, i+1)
    ax.scatter(X[i][:,0], X[i][:,1], c=y[i])
'''

'''
#GaussianMixture
y = []
for c, x in zip(classes, X):
    model = mixture.GaussianMixture(n_components=c, covariance_type="full")
    model.fit(x)
    if hasattr(model, 'labels_'):
        y.append(model.labels_.astype(np.int))
    else:
        y.append(model.predict(x))

fig = plt.figure(figsize=(27,9))
fig.suptitle('GaussianMixture', fontsize=48)
for i in range(6):
    ax = plt.subplot(2, 3, i+1)
    ax.scatter(X[i][:,0], X[i][:,1], c=y[i])
'''

'''
#OPTICS
y = []
for c, x in zip(classes, X):
    model = cluster.OPTICS(min_samples=20, xi=0.05, min_cluster_size=0.1)
    model.fit(x)
    if hasattr(model, 'labels_'):
        y.append(model.labels_.astype(np.int))
    else:
        y.append(model.predict(x))

fig = plt.figure(figsize=(27,9))
fig.suptitle('OPTICS', fontsize=48)
for i in range(6):
    ax = plt.subplot(2, 3, i+1)
    ax.scatter(X[i][:,0], X[i][:,1], c=y[i])
'''

'''
#DBSCAN
y = []
eps = [0.3, 0.3 , 0.3, 0.3, 0.15, 0.18]
for c, x, e in zip(classes, X, eps):
    model = cluster.DBSCAN(eps=e)
    model.fit(x)
    if hasattr(model, 'labels_'):
        y.append(model.labels_.astype(np.int))
    else:
        y.append(model.predict(x))

fig = plt.figure(figsize=(27,9))
fig.suptitle('DBSCAN', fontsize=48)
for i in range(6):
    ax = plt.subplot(2, 3, i+1)
    ax.scatter(X[i][:,0], X[i][:,1], c=y[i])
'''



