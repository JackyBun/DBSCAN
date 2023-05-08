import hdbscan

class HDBSCANClusterer:
    def __init__(self, min_cluster_size, min_samples=None):
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples

    def fit_predict(self, X):
        hdbscan_clusterer = hdbscan.HDBSCAN(min_cluster_size=self.min_cluster_size, min_samples=self.min_samples)
        hdbscan_clusterer.fit(X)
        return hdbscan_clusterer.labels_
