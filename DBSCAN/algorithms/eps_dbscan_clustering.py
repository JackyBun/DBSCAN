import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors

class EpsilonDBSCANClusterer:
    def __init__(self, min_samples):
        self.min_samples = min_samples

    def fit_predict(self, X):
        neigh = NearestNeighbors(n_neighbors=self.min_samples)
        nbrs = neigh.fit(X)
        distances, indices = nbrs.kneighbors(X)
        distances = np.sort(distances, axis=0)
        distances = distances[:,1]
        eps = np.median(distances)
        dbscan = DBSCAN(eps=eps, min_samples=self.min_samples)
        dbscan.fit(X)
        return dbscan.labels_
