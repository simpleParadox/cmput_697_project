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
from hdbscan import HDBSCAN

from sklearn.manifold import TSNE
import seaborn as sns

from sklearn.metrics import silhouette_score
from sklearn.metrics import adjusted_rand_score
from sklearn import metrics


class Clustering:

    def __init__(self, n_clusters=None, eps=0.5, min_samples=5, min_cluster_size=5, metric="euclidean", clustering_class="partitioning", seed=42):
        """
        Initialize the clustering algorithms.
        :param n_clusters: for kmeans and hierarchical clustering.
        :param eps: for dbscan.
        :param min_samples: for optics.
        :param metric: for the metric used to compute the distances between the pairs of instances.
        """
        self.clustering_class = clustering_class
        if clustering_class == "other":
            self.alg1 = KMeans(n_clusters=n_clusters, random_state=seed)
            self.alg2 = AgglomerativeClustering(n_clusters=n_clusters, metric=metric, linkage="single")
        elif clustering_class == "density":
            # The metric is used only for the density based clustering algorithm.
            self.alg1 = DBSCAN(eps=eps, min_samples=min_samples, metric=metric)
            if metric != 'cosine':
                self.alg2 = HDBSCAN(min_cluster_size=min_cluster_size, metric=metric)
            else:
                print("HDBSCAN does not support cosine metric. Using DBSCAN instead.")
                self.alg2 = DBSCAN(eps=eps, min_samples=min_samples, metric=metric)

    def train(self, X):
        self.alg1.fit(X)
        self.alg2.fit(X)

    def plot_clusters(self, embedding_data, true_labels, cluster_labels, algorithm='KMeans'):
        tsne = TSNE(n_components=2)
        tsne_result = tsne.fit_transform(embedding_data)  # This is after dimensionality reduction.

        # Get the components.
        x = tsne_result[:, 0]
        y = tsne_result[:, 1]

        # Create a dataframe for easy plotting using seaborn.
        df = pd.DataFrame({'x': x, 'y': y, 'true_label': true_labels, 'cluster_label': cluster_labels})

        # Plot using seaborn.
        plt.figure(figsize=(8, 8))
        ax = sns.scatterplot(x='x', y='y', hue='cluster_label', data=df, palette=sns.color_palette("hls", 10),
                             legend="full")
        # ax  = sns.scatterplot()  # Fill this out to plot the centroids.
        plt.savefig(f"figures/{algorithm}_cluster_labels.png")

        plt.clf()
        plt.figure(figsize=(8, 8))
        ax = sns.scatterplot(x='x', y='y', hue='true_label', data=df, palette=sns.color_palette("hls", 10),
                             legend="full")
        # ax  = sns.scatterplot()  # Fill this out to plot the centroids.
        plt.savefig(f"figures/{algorithm}_true_labels.png")

    def remove_outliers(self, X, y):
        outliers1 = np.where(self.alg1.labels_ == -1)[0]
        outliers2 = np.where(self.alg2.labels_ == -1)[0]
        # Remove the outliers from the data.
        X1 = np.delete(X, outliers1, axis=0)
        X2 = np.delete(X, outliers2, axis=0)
        # Remove the outliers from the predicted labels.

        y1 = np.delete(self.alg1.labels_, outliers1, axis=0)
        y2 = np.delete(self.alg2.labels_, outliers2, axis=0)

        # Remove the outliers from the ground truth labels.
        y_true1 = np.delete(y, outliers1, axis=0)
        y_true2 = np.delete(y, outliers2, axis=0)


        return X1, X2, y1, y2, y_true1, y_true2
    def validate(self, X, y=None, method="internal", plot_clusters=True):
        """
        Validate the clustering performance using some sort of internal and external validation.
        Supports only silhouette score for internal validation and adjusted rand index for external validation.
        :param X: data
        :param y: labels after clustering (for internal validation) or true labels (for external validation)
        :param method: 'internal' or 'external'
        :return: score
        """
        X1, X2, y1, y2, y_true1, y_true2 = self.remove_outliers(X, y)
        if method == "internal":
            # assert y == None, "y should be None for internal validation."
            # Obtain the indices where the label value is -1.
            # These are the outliers.

            # X1, X2, y1, y2 = self.remove_outliers(X, y)

            # outliers1 = np.where(self.alg1.labels_ == -1)[0]
            # outliers2 = np.where(self.alg2.labels_ == -1)[0]
            # # Remove the outliers from the data.
            # X1 = np.delete(X, outliers1, axis=0)
            # X2 = np.delete(X, outliers2, axis=0)
            # # Remove the outliers from the labels.
            #
            # y1 = np.delete(self.alg1.labels_, outliers1, axis=0)
            # y2 = np.delete(self.alg2.labels_, outliers2, axis=0)
            # print("labels1 :", y1)
            # print("labels2:", y2)
            alg1_score = silhouette_score(X1, y1)
            alg2_score = silhouette_score(X2, y2)
            return alg1_score, alg2_score
        elif method == "external":
            # assert y != None, "y should not be None for external validation."

            alg1_score = adjusted_rand_score(y1, y_true1)
            alg2_score = adjusted_rand_score(y2, y_true2)
            return alg1_score, alg2_score
        elif method == 'purity':
            # First remove the ourliers.
            # Taken from the sklearn answer computing purity.
            # 'y' is the ratings variable.
            # # compute contingency matrix (also called confusion matrix)
            # outliers1 = np.where(self.alg1.labels_ == -1)[0]
            # outliers2 = np.where(self.alg2.labels_ == -1)[0]
            #
            # y1 = np.delete(self.alg1.labels_, outliers1, axis=0)
            # y2 = np.delete(self.alg2.labels_, outliers2, axis=0)

            y_pred1 = y1
            y_pred2 = y2
            contingency_matrix1 = metrics.cluster.contingency_matrix(y_true1, y_pred1)
            contingency_matrix2 = metrics.cluster.contingency_matrix(y_true2, y_pred2)
            # return purity
            purity1 = np.sum(np.amax(contingency_matrix1, axis=0)) / np.sum(contingency_matrix1)
            purity2 = np.sum(np.amax(contingency_matrix2, axis=0)) / np.sum(contingency_matrix2)
            return purity1, purity2
        else:
            raise ValueError("Method not supported.")
