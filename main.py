# Custom python files.
from preprocess import load_data, store_embeddings
from models import Clustering
import numpy as np





# Load the data
reviews, ratings = load_data()


# Get the indices where the ratings are not nan.
good_indices = np.where(np.isnan(ratings) == False)[0]
ratings = np.array(ratings)[good_indices]
# Get the embeddings.
# embeddings = store_embeddings(reviews, model_name="bert", store_path="/Users/simpleparadox/PycharmProjects/cmput_697/embeds/bert_avg.npz")


iterations = 50
possible_clusters = [3, 5]
k_means_internal = np.zeros((iterations, len(possible_clusters)))
agglomerative_internal = np.zeros((iterations, len(possible_clusters)))
k_means_external = np.zeros((iterations, len(possible_clusters)))
agglomerative_external = np.zeros((iterations, len(possible_clusters)))

possible_eps = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
dbscan_internal = np.zeros((iterations, len(possible_eps)))
dbscan_external = np.zeros((iterations, len(possible_eps)))
optics_internal = np.zeros((iterations, len(possible_eps)))
optics_external = np.zeros((iterations, len(possible_eps)))



for i in range(iterations):
    clustering_class = 'partitioning'
    # Load the embeddings.
    embed_names = ["bert_avg", "bert_embeddings", "w2v_embeddings"]
    for embed in embed_names:
        embedding = np.load(f"/Users/simpleparadox/PycharmProjects/cmput_697/embeds/{embed}.npz")['arr_0']
        embedding = embedding[good_indices]
        # Run the clustering algorithms on the selected embeddings.
        if clustering_class == 'partitioning':
            for c_i, n_cluster in enumerate(possible_clusters):
                print(f"Number of clusters: {n_cluster}")
                clustering = Clustering(n_clusters=n_cluster, eps=None, min_samples=5, metric="euclidean")
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
                kmeans_score, agglomerative_score = clustering.validate(reviews, ratings, method='external')
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
                dbscan_score, optics_score = clustering.validate(reviews, None, method='internal')  # NOTE: y (ratings) is not used for internal validation. Used for consistency.
                print("DBSCAN silhouette score: ", dbscan_score)
                dbscan_internal[i, eps_i] = dbscan_score
                print("OPTICS silhouette score: ", optics_score)
                optics_internal[i, eps_i] = optics_score

                # Do external validation.
                print("External validation for ", embed)
                dbscan_score, optics_score = clustering.validate(reviews, ratings, method='external')  # NOTE: y (ratings) is not used for internal validation. Used for consistency.
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