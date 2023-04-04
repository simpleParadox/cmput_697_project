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


iterations = 50
possible_clusters = [5] # range(3, 20, 2)
k_means_internal = np.zeros((iterations, len(possible_clusters)))
agglomerative_internal = np.zeros((iterations, len(possible_clusters)))
k_means_external = np.zeros((iterations, len(possible_clusters)))
agglomerative_external = np.zeros((iterations, len(possible_clusters)))

# Really low values for values of eps under 10.0. NOTE: a value of 70.0 for DBSCAN leads to only one cluster.
# Therefore, the minimum requirement is to have at least 2 clusters.
# possible_eps = [10.0, 20.0, 30.0, 40.0, 50.0]
# possible_eps = [0.5, 1.0, 5.0, 10.0, 15.0] # Stop when the number of labels is less than 2. Not using 20.0. but there for compatibility.
possible_eps = [10.0]

possible_min_cluster_size = [5, 10, 20, 25, 30]
dbscan_internal = np.zeros((iterations, len(possible_eps)))
dbscan_external = np.zeros((iterations, len(possible_eps)))
hdbscan_internal = np.zeros((iterations, len(possible_eps)))
hdbscan_external = np.zeros((iterations, len(possible_eps)))


if possible_clusters[0] == 3:
    # Relabel the ratings if the number of selected clusters is 3, else leave them unchanged.
    ratings[ratings < 3] = 0
    ratings[ratings == 3] = 1
    ratings[ratings > 3] = 2


labels_dbscan = {}
labels_hdbscan = {}

labels_kmeans = {}
labels_agglomerative = {}


for i in range(iterations):
    print("Iteration: ", i)
    clustering_class = 'partitioning'
    # Load the embeddings.
    # embed_names = ["bert_avg"]
    # embed_names = ["bert_embeddings"]
    embed_names = ["w2v_embeddings"]
    for embed in embed_names:
        embedding = np.load(f"embeds/{embed}.npz", allow_pickle=True)['arr_0']
        embedding = embedding[good_indices]

        # Apply standard scalar on the embeddings.
        scaler = StandardScaler()
        embedding = scaler.fit_transform(embedding)

        inertia = []

        # Run the clustering algorithms on the selected embeddings.
        if clustering_class == 'partitioning':
            for c_i, n_cluster in enumerate(possible_clusters):
                print(f"Number of clusters: {n_cluster}")

                clustering = Clustering(n_clusters=n_cluster, eps=None, min_samples=5, metric="euclidean", clustering_class=clustering_class, seed=i)
                clustering.train(embedding)

                # Do internal validation.
                print("Internal validation for ", embed)
                kmeans_score, agglomerative_score = clustering.validate(embedding, None, method='internal')  # NOTE: y (ratings) is not used for internal validation. Used for consistency.
                print("Kmeans silhouette score: ", kmeans_score)
                k_means_internal[i, c_i] = kmeans_score
                print("Hierarchical silhouette score: ", agglomerative_score)
                agglomerative_internal[i, c_i] = agglomerative_score

                labels_kmeans[n_cluster] = clustering.alg1.labels_
                labels_agglomerative[n_cluster] = clustering.alg2.labels_

                # Store the inertia scores for k-means clustering.
                inertia.append(clustering.alg1.inertia_)



                # Do external validation.
                print("External validation for ", embed)
                # plot_clusters(embedding, ratings, clustering.alg1.labels_, 'KMeans', 3)
                # The check that we have to make here is if the predicted and the true labels are the same. Plotting them will help.
                kmeans_score, agglomerative_score = clustering.validate(embedding, ratings, method='external')  # NOTE: 'embeddings' is not used here, kept for consistency.
                print("Kmeans adjusted_rand_index: ", kmeans_score)
                k_means_external[i, c_i] = kmeans_score
                print("Hierarchical adjusted_rand_index: ", agglomerative_score)
                agglomerative_external[i, c_i] = agglomerative_score

                # plot_clusters(embedding_data=embedding, true_labels=ratings, cluster_labels=clustering.alg1.labels_, algorithm=f'K_Means - {embed}_{n_cluster}_clusters', num_clusters=n_cluster)
                # plot_clusters(embedding_data=embedding, true_labels=ratings, cluster_labels=clustering.alg2.labels_, algorithm=f'K_Means - {embed}_{n_cluster}_clusters', num_clusters=n_cluster)
        else:
            # possible_eps = [2, 5, 10, 15, 20]  # Redefining this for the hdbscan.

            for eps_i, eps in enumerate(possible_eps):
                print(f"Epsilon value: {eps}")
                clustering = Clustering(n_clusters=None, eps=5, min_cluster_size=10, min_samples=5, metric="euclidean", clustering_class=clustering_class)
                clustering.train(embedding)
                print("Unique cluster labels: ", np.unique(clustering.alg1.labels_))


                # Store the labels for DBSCAN.
                labels_dbscan[eps] = clustering.alg1.labels_

                # Store the labels for HDBSCAN.
                labels_hdbscan[eps] = clustering.alg2.labels_

                # Do internal validation.
                print("Internal validation for ", embed)
                dbscan_score, hdbscan_score = clustering.validate(embedding, None, method='internal')  # NOTE: y (ratings) is not used for internal validation. Used for consistency.
                print("DBSCAN silhouette score: ", dbscan_score)
                # dbscan_internal[i, eps_i] = dbscan_score

                print("HDBSCAN silhouette score: ", hdbscan_score)
                hdbscan_internal[i, eps_i] = hdbscan_score

                # # Do external validation.
                # print("External validation for ", embed)
                # # dbscan_score, hdbscan_score = clustering.validate(embedding, ratings, method='external')  # NOTE: 'embeddings' is not used here.
                # print("DBSCAN adjusted rand index: ", dbscan_score)
                # # dbscan_external[i, eps_i] = dbscan_score
                # print("HDBSCAN adjusted rand index: ", hdbscan_score)
                # # hdbscan_external[i, eps_i] = hdbscan_score
    print("------------------------------------------------------------------------------------------------------------------")
    print("KMeans internal: ", k_means_internal)
    print("Agglomerative internal: ", agglomerative_internal)
    print("DBSCAN internal: ", dbscan_internal)
    print("HDBSCAN internal: ", hdbscan_internal)
    print("------------------------------------------------------------------------------------------------------------------")
    print("KMeans external: ", k_means_external)
    print("Agglomerative external: ", agglomerative_external)
    print("DBSCAN external: ", dbscan_external)
    print("HDBSCAN external: ", hdbscan_external)

# Save the results.
# First create a dictionary to store the scores and the labels.
# results_dbscan = {}
# results_hdbscan = {}
# results_dbscan['dbscan_internal'] = dbscan_internal
# results_hdbscan['hdbscan_internal'] = hdbscan_internal
# results_dbscan['dbscan_labels'] = labels_dbscan
# results_hdbscan['hdbscan_labels'] = labels_hdbscan

results_kmeans = {}
results_agglomerative = {}
results_kmeans['k_means_internal'] = k_means_internal
results_agglomerative['agglomerative_internal'] = agglomerative_internal
results_kmeans['k_means_external'] = k_means_external
results_agglomerative['agglomerative_external'] = agglomerative_external
results_kmeans['k_means_labels'] = labels_kmeans
results_agglomerative['agglomerative_labels'] = labels_agglomerative


# np.savez_compressed(f"results/dbscan_internal_{embed}_{possible_clusters[0]}.npz", np.array(results_dbscan))
# np.savez_compressed(f"results/dbscan_external_{embed}_{possible_clusters[0]}.npz", dbscan_external)
# np.savez_compressed(f"results/hdbscan_internal_{embed}_{possible_clusters[0]}.npz", results_hdbscan)
# np.savez_compressed(f"results/hdbscan_external_{embed}_{possible_clusters[0]}.npz", hdbscan_external)
# Calculate the means of the internal and external scores for each algorithm and

# Save the results for the partitioning algorithms.
np.savez_compressed(f"results/k_means_external_{embed}_{possible_clusters[0]}_50_iters_fixed_seed.npz", np.array(k_means_external))
# np.savez_compressed(f"results/agglomerative_results_{embed}_{possible_clusters[0]}.npz", np.array(results_agglomerative))




# TODO: Have to select the number of clusters for the partitioning algorithms. Use the elbow method to select the number of clusters.

# NOTE: Clsuters can be automatically selected by the density based algorithms but not by the partitioning algorithms. So a good question to investigate would be to see if the number of clusters selected by the partitioning algorithms is similar to the number of clusters selected by the density based algorithms.
# This also means that it would take some thinking of how to divide the ratings labels.
# One option is to divide the ratings into 5 clusters.
# Another option is to divide the ratings into 3 clusters. # star ratings will be labelled as neutral, 1 and 2 stars will be labelled as negative and 4 and 5 stars will be labelled as positive.
# Question to ask?: Choosing 3 or 5 clusters will have an effect on the performance because it may be the case that ratings 4/5 might be clustered in the positive group together but ratings 1/2 might be clustered in the negative group together.
# One hypothesis is as follows:
# 1. The performance of the algorithms are better when there are 5 clusters instead of 3 clusters.
# 2. The performance of the algorithms are better when the ratings are divided into 3 clusters instead of 5 clusters.



# from sklearn.neighbors import NearestNeighbors
#
# neighbors = NearestNeighbors(n_neighbors=50)
# neighbors_fit = neighbors.fit(embedding)
# distances, indices = neighbors_fit.kneighbors(embedding)
# plt.clf()
# distances = np.sort(distances, axis=0)
# distances = distances[:,1]
# plt.plot(distances)
# plt.show()
#
#
#
from scipy.stats import sem


kmeans_external_bert_avg_3 = np.load("results/k_means_external_bert_avg_3_50_iters_fixed_seed.npz")['arr_0']
kmeans_external_bert_avg_5 = np.load("results/k_means_external_bert_avg_5_50_iters_fixed_seed.npz")['arr_0']
kmeans_external_bert_embeddings_3 = np.load("results/k_means_external_bert_embeddings_3_50_iters_fixed_seed.npz")['arr_0']
kmeans_external_bert_embeddings_5 = np.load("results/k_means_external_bert_embeddings_5_50_iters_fixed_seed.npz")['arr_0']
kmeans_external_w2v_embeddings_3 = np.load("results/k_means_external_w2v_embeddings_3_50_iters_fixed_seed.npz")['arr_0']
kmeans_external_w2v_embeddings_5 = np.load("results/k_means_external_w2v_embeddings_5_50_iters_fixed_seed.npz")['arr_0']


# Create a dataframe to store the results.
# results = pd.DataFrame(columns=['bert_avg_3', 'bert_avg_5', 'bert_embeddings_3', 'bert_embeddings_5', 'w2v_embeddings_3', 'w2v_embeddings_5'])
# results['bert_avg_3'] = kmeans_external_bert_avg_3
# results['bert_avg_5'] = kmeans_external_bert_avg_5
# results['bert_embeddings_3'] = kmeans_external_bert_embeddings_3
# results['bert_embeddings_5'] = kmeans_external_bert_embeddings_5
# results['w2v_embeddings_3'] = kmeans_external_w2v_embeddings_3
# results['w2v_embeddings_5'] = kmeans_external_w2v_embeddings_5


# Calculate the mean and standard error of the mean for each algorithm.
embedding_type = ['BERT - Average', 'BERT - CLS', 'Word2Vec - Average']

results_dict = {
    '3': [np.mean(kmeans_external_bert_avg_3), np.mean(kmeans_external_bert_embeddings_3), np.mean(kmeans_external_w2v_embeddings_3)],
    '5': [np.mean(kmeans_external_bert_avg_5), np.mean(kmeans_external_bert_embeddings_5), np.mean(kmeans_external_w2v_embeddings_5)]
}

x = np.arange(len(embedding_type))  # the label locations
width = 0.35  # the width of the bars
multiplier = 0
plt.clf()
sns.set_style("whitegrid")
fig, ax = plt.subplots(layout='tight')

for attribute, measurement in results_dict.items():
    offset = width * multiplier
    rects = ax.bar(x + offset, measurement, width, label=attribute)
    ax.bar_label(rects, padding=3)
    multiplier += 1

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Adjusted Rand Index', fontsize=14)
ax.set_title('KMeans - External Validation for 3 and 5 clusters')
ax.set_xticks(x + width, embedding_type, fontsize=12)
ax.legend(loc='upper left', ncols=3, title='Number of clusters')
ax.set_xlabel("Embedding type", fontsize=14)
plt.savefig('plots/score_plots/kmeans_external_validation.pdf')
# plt.show()



# # Calculate the mean and standard error of the mean.
# k_means_mean = np.mean(k_means_internal, axis=0)
# k_means_sem = sem(k_means_internal, axis=0)
#
#
# # Plot the inertia for the k-means algorithm.
# plt.clf()
# sns.set_style("whitegrid")
# plt.plot(possible_clusters, k_means_mean)
# plt.fill_between(possible_clusters, k_means_mean - k_means_sem, k_means_mean + k_means_sem, alpha=0.1)
# # plt.plot(possible_clusters, results_kmeans['k_means_external'].reshape(-1))
# plt.title('Silhouette score for k-means')
# plt.xticks(possible_clusters)
# plt.xlabel('Number of clusters')
# plt.ylabel('Silhouette score')
# plt.show()
