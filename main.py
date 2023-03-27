# Custom python files.
from preprocess import load_data, store_embeddings
from models import Clustering

# General python data science packages.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE

def plot_clusters(embedding_data, true_labels, cluster_labels, algorithm='KMeans', num_clusters=3):
    tsne = TSNE(n_components=2)
    tsne_result = tsne.fit_transform(embedding_data)  # This is after dimensionality reduction.

    # Get the components.
    x = tsne_result[:, 0]
    y = tsne_result[:, 1]

    # Create a dataframe for easy plotting using seaborn.
    df = pd.DataFrame({'x': x, 'y': y, 'true_label': true_labels, 'cluster_label': cluster_labels})

    # Plot using seaborn.
    plt.figure(figsize=(8, 8))
    ax = sns.scatterplot(x='x', y='y', hue='cluster_label', data=df, palette=sns.color_palette("hls", num_clusters),
                         legend="full")
    # ax  = sns.scatterplot()  # Fill this out to plot the centroids.
    plt.savefig(f"figures/{algorithm}_cluster_labels.png")

    plt.clf()
    plt.figure(figsize=(8, 8))
    ax = sns.scatterplot(x='x', y='y', hue='true_label', data=df, palette=sns.color_palette("hls", num_clusters),
                         legend="full")
    # ax  = sns.scatterplot()  # Fill this out to plot the centroids.
    plt.savefig(f"figures/{algorithm}_true_labels.png")




# Load the data
reviews, ratings = load_data()


# Get the indices where the ratings are not nan.
good_indices = np.where(np.isnan(ratings) == False)[0]
ratings = np.array(ratings)[good_indices]
ratings -= 1 # To be in line with the labels assigned by the clustering algorithm.

# The following three lines are when the number of clusters is 3.


# Get the embeddings.
# embeddings = store_embeddings(reviews, model_name="bert", store_path="/Users/simpleparadox/PycharmProjects/cmput_697/embeds/bert_avg.npz")


iterations = 1
possible_clusters = [3]
k_means_internal = np.zeros((iterations, len(possible_clusters)))
agglomerative_internal = np.zeros((iterations, len(possible_clusters)))
k_means_external = np.zeros((iterations, len(possible_clusters)))
agglomerative_external = np.zeros((iterations, len(possible_clusters)))

possible_eps = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
dbscan_internal = np.zeros((iterations, len(possible_eps)))
dbscan_external = np.zeros((iterations, len(possible_eps)))
optics_internal = np.zeros((iterations, len(possible_eps)))
optics_external = np.zeros((iterations, len(possible_eps)))


if possible_clusters[0] == 3:
    # Relabel the ratings if the number of selected clusters is 3, else leave them unchanged.
    ratings[ratings < 3] = 0
    ratings[ratings == 3] = 1
    ratings[ratings > 3] = 2


for i in range(iterations):
    clustering_class = 'partitioning'
    # Load the embeddings.
    embed_names = ["bert_avg", "bert_embeddings", "w2v_embeddings"]
    for embed in embed_names:
        embedding = np.load(f"embeds/{embed}.npz")['arr_0']
        embedding = embedding[good_indices]

        # Apply standard scalar on the embeddings.
        scaler = StandardScaler()
        embedding = scaler.fit_transform(embedding)

        # Run the clustering algorithms on the selected embeddings.
        if clustering_class == 'partitioning':
            for c_i, n_cluster in enumerate(possible_clusters):
                print(f"Number of clusters: {n_cluster}")

                clustering = Clustering(n_clusters=n_cluster, eps=None, min_samples=5, metric="cosine")
                clustering.train(embedding)

                # Do internal validation.
                print("Internal validation for ", embed)
                kmeans_score, agglomerative_score = clustering.validate(embedding, None, method='internal')  # NOTE: y (ratings) is not used for internal validation. Used for consistency.
                print("Kmeans silhouette score: ", kmeans_score)
                k_means_internal[i, c_i] = kmeans_score
                print("Hierarchical silhouette score: ", agglomerative_score)
                agglomerative_internal[i, c_i] = agglomerative_score


                # Do external validation.
                print("External validation for ", embed)
                plot_clusters(embedding, ratings, clustering.alg1.labels_, 'KMeans', 3)
                # The check that we have to make here is if the predicted and the true labels are the same. Plotting them will help.
                kmeans_score, agglomerative_score = clustering.validate(embedding, ratings, method='external')  # NOTE: 'embeddings' is not used here.
                print("Kmeans adjusted_rand_index: ", kmeans_score)
                k_means_external[i, c_i] = kmeans_score
                print("Hierarchical adjusted_rand_index: ", agglomerative_score)
                agglomerative_external[i, c_i] = agglomerative_score
        else:
            for eps_i, eps in enumerate(possible_eps):
                print(f"Epsilon value: {eps}")
                clustering = Clustering(n_clusters=None, eps=eps, min_samples=5, metric="euclidean")
                clustering.train(embedding)

                # Do internal validation.
                print("Internal validation for ", embed)
                dbscan_score, optics_score = clustering.validate(embedding, None, method='internal')  # NOTE: y (ratings) is not used for internal validation. Used for consistency.
                print("DBSCAN silhouette score: ", dbscan_score)
                dbscan_internal[i, eps_i] = dbscan_score
                print("OPTICS silhouette score: ", optics_score)
                optics_internal[i, eps_i] = optics_score

                # Do external validation.
                print("External validation for ", embed)
                dbscan_score, optics_score = clustering.validate(embedding, ratings, method='external')  # NOTE: 'embeddings' is not used here.
                print("DBSCAN adjusted rand index: ", dbscan_score)
                dbscan_external[i, eps_i] = dbscan_score
                print("OPTICS adjusted rand index: ", optics_score)
                optics_external[i, eps_i] = optics_score


# Calculate the means of the internal and external scores for each algorithm and



# TODO: Have to select the number of clusters for the partitioning algorithms. Use the elbow method to select the number of clusters.

# NOTE: Clsuters can be automatically selected by the density based algorithms but not by the partitioning algorithms. So a good question to investigate would be to see if the number of clusters selected by the partitioning algorithms is similar to the number of clusters selected by the density based algorithms.
# This also means that it would take some thinking of how to divide the ratings labels.
# One option is to divide the ratings into 5 clusters.
# Another option is to divide the ratings into 3 clusters. # star ratings will be labelled as neutral, 1 and 2 stars will be labelled as negative and 4 and 5 stars will be labelled as positive.
# Question to ask?: Choosing 3 or 5 clusters will have an effect on the performance because it may be the case that ratings 4/5 might be clustered in the positive group together but ratings 1/2 might be clustered in the negative group together.
# One hypothesis is as follows:
# 1. The performance of the algorithms are better when there are 5 clusters instead of 3 clusters.
# 2. The performance of the algorithms are better when the ratings are divided into 3 clusters instead of 5 clusters.