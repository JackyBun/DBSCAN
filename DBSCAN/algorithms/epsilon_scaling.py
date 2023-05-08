import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors

class EpsilonScalingDBSCAN:
    def __init__(self, p, min_samples):
        self.p = p
        self.min_samples = min_samples
    
    def fit_predict(self, X):
        # Compute the k-nearest neighbors for each point
        nn = NearestNeighbors(n_neighbors=self.min_samples, p=self.p)
        nn.fit(X)
        kdistances, _ = nn.kneighbors(X)
        
        # Sort k-distances in ascending order
        kdistances = np.sort(kdistances, axis=0)
        kdistances = kdistances[:, -1::-1]
        
        # Calculate the eps values for each point
        eps_values = np.zeros_like(kdistances)
        for i in range(len(kdistances)):
            eps_values[i] = kdistances[i][np.argmax(np.diff(kdistances[i]))]
        
        # Perform DBSCAN with the computed eps values
        eps = eps_values.min() if eps_values.min() > 0 else 0.1
        dbscan = DBSCAN(eps=eps, min_samples=self.min_samples)

        labels = dbscan.fit_predict(X)
        return labels
