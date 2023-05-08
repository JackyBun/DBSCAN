from sklearn.cluster import DBSCAN

class DBSCANClusterer:
    def __init__(self, eps, min_samples):
        self.eps = eps
        self.min_samples = min_samples
        self.clusterer = DBSCAN(eps=eps, min_samples=min_samples)

    def fit_predict(self, X):
        self.clusterer.fit(X)
        return self.clusterer.labels_
