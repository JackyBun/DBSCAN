import numpy as np
from sklearn.cluster import DBSCAN
from joblib import Parallel, delayed

class ParallelDBSCANClusterer:
    def __init__(self, eps, min_samples, n_jobs=-1):
        self.eps = eps
        self.min_samples = min_samples
        self.n_jobs = n_jobs

    def fit_predict(self, X):
        def dbscan_chunk(X_chunk):
            dbscan = DBSCAN(eps=self.eps, min_samples=self.min_samples)
            dbscan.fit(X_chunk)
            return dbscan.labels_

        labels_chunks = Parallel(n_jobs=self.n_jobs)(
            delayed(dbscan_chunk)(X_chunk) for X_chunk in np.array_split(X, self.n_jobs)
        )
        labels = np.concatenate(labels_chunks)
        return labels
