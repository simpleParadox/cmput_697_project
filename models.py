"""
Create a class with all the clustering algorithms inside a single class.
Define a train function to fit all the clustering algorithms on the data.
Define a predict function to predict the clustering performance using some sort of internal and external validation.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN
from sklearn.cluster import OPTICS

from sklearn.metrics import silhouette_score
from sklearn.metrics import adjusted_rand_score



class Clustering:

    def __init__(self, n_clusters=None, eps=0.5, min_samples=5, metric="euclidean", clustering_class="partitioning"):
        """
        Initialize the clustering algorithms.
        :param n_clusters: for kmeans and hierarchical clustering.
        :param eps: for dbscan.
        :param min_samples: for optics.
        :param metric: for the metric used to compute the distances between the pairs of instances.
        """
        self.clustering_class = clustering_class
        if clustering_class == "partitioning":
            self.alg1 = KMeans(n_clusters=n_clusters, random_state=42)
            self.alg2 = AgglomerativeClustering(n_clusters=n_clusters)
        else:
            self.alg1 = DBSCAN(eps=eps, min_samples=min_samples, metric=metric)
            self.alg2 = OPTICS(eps=eps, min_samples=min_samples, metric=metric)

    def train(self, X):
        self.alg1.fit(X)
        self.alg2.fit(X)


    def validate(self, X, y=None, method="internal"):
        """
        Validate the clustering performance using some sort of internal and external validation.
        Supports only silhouette score for internal validation and adjusted rand index for external validation.
        :param X: data
        :param y: labels after clustering (for internal validation) or true labels (for external validation)
        :param method: 'internal' or 'external'
        :return: score
        """
        if method == "internal":
            # assert y == None, "y should be None for internal validation."
            alg1_score = silhouette_score(X, self.alg1.labels_)
            alg2_score = silhouette_score(X, self.alg2.labels_)
            return alg1_score, alg2_score
        elif method == "external":
            assert y != None, "y should not be None for external validation."
            alg1_score = adjusted_rand_score(y, self.alg1.labels_)
            alg2_score = adjusted_rand_score(y, self.alg2.labels_)
            return alg1_score, alg2_score
        else:
            raise ValueError("Method not supported.")
