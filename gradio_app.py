import gradio as gr
import numpy as np

from preprocess import load_data, store_embeddings
from models import Clustering

# General python data science packages.
import pandas as pd
import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE


from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN
from sklearn.cluster import OPTICS
from hdbscan import HDBSCAN

from sklearn.metrics import silhouette_score
from sklearn.metrics import adjusted_rand_score

def load_review_and_ratings():
    # Load the data
    reviews, ratings = load_data()

    # Get the indices where the ratings are not nan.
    good_indices = np.where(np.isnan(ratings) == False)[0]
    ratings = np.array(ratings)[good_indices]
    ratings -= 1

    return reviews, ratings, good_indices

def get_df_for_plotting(embedding_data, true_labels, cluster_labels):

    tsne = TSNE(n_components=2)
    tsne_result = tsne.fit_transform(embedding_data)  # This is after dimensionality reduction.

    # Get the components.
    x = tsne_result[:, 0]
    y = tsne_result[:, 1]

    # Create a dataframe for easy plotting using seaborn.
    df = pd.DataFrame({'x': x, 'y': y, 'true_label': true_labels, 'cluster_label': cluster_labels})
    return df, df




def run_clustering(embedding_type, clustering_algorithm, seed, n_clusters, eps, min_samples, min_cluster_size, metric):
    # Write a bunch of if else conditions
    print("Running clustering")
    print("Number of clusters: ", n_clusters)
    n_clusters = int(n_clusters)
    min_samples = int(min_samples)
    min_cluster_size = int(min_cluster_size)

    reviews, ratings, good_indices = load_review_and_ratings()

    if n_clusters == 3:
        print("Relabeling the ratings")
        # Relabel the ratings if the number of selected clusters is 3, else leave them unchanged.
        ratings[ratings < 3] = 0
        ratings[ratings == 3] = 1
        ratings[ratings > 3] = 2


    embedding = None
    model = None

    if embedding_type == 'Word2Vec - Average':
        print("Loading Word2Vec - Average")
        embedding = np.load(f"embeds/w2v_embeddings.npz")['arr_0']
        embedding = embedding[good_indices]

    elif embedding_type == 'BERT - Average':
        print("Loading BERT Average embeddings")
        embedding = np.load(f"embeds/bert_avg.npz")['arr_0']
        embedding = embedding[good_indices]

    elif embedding_type == 'BERT - CLS':
        print("Loading BERT CLS embeddings")
        embedding = np.load(f"embeds/bert_embeddings.npz")['arr_0']
        embedding = embedding[good_indices]

    assert embedding is not None

    # Mean center the data.
    scaler = StandardScaler()
    embedding = scaler.fit_transform(embedding)

    if clustering_algorithm == 'KMeans':
        print("Running KMeans")
        model = KMeans(n_clusters=n_clusters, random_state=int(seed))
        model.fit(embedding)

    elif clustering_algorithm == 'Agglomerative Hierarchical - single linkage':
        print("Running Agglomerative Hierarchical - single linkage")
        model = AgglomerativeClustering(n_clusters=n_clusters, metric=metric, linkage='single')
        model.fit(embedding)

    elif clustering_algorithm == 'DBSCAN':
        print("Running DBSCAN")
        model = DBSCAN(eps=eps, min_samples=min_samples, metric=metric)
        model.fit(embedding)

    elif clustering_algorithm == 'HDBSCAN':
        print("Running HDBSCAN")
        if metric == 'cosine':
            raise gr.Error("HDBSCAN does not support cosine similarity. Please choose another metric.")
        model = HDBSCAN(min_cluster_size=min_cluster_size, metric=metric)
        model.fit(embedding)


    # Evaluate the model.
    assert model is not None
    # Remove the -1 labels.
    outliers = np.where(model.labels_ == -1)[0]
    # Remove the outliers from the data.
    X = np.delete(embedding, outliers, axis=0)
    # Remove the outliers from the labels.

    y = np.delete(model.labels_, outliers, axis=0)
    silhouette = silhouette_score(X, y)
    print("Silhouette score: ", silhouette)
    ari = adjusted_rand_score(ratings, model.labels_)

    return silhouette, ari#, get_df_for_plotting(embedding, ratings, model.labels_)


with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.HTML('<h1>Choose Embedding type and Clustering Algorithm</h1>')
    with gr.Row():
        with gr.Column():
            embedding_type = gr.Radio(['Word2Vec - Average', 'BERT - Average', 'BERT - CLS'], label='Embedding type', info='Choose the embedding type.')
        with gr.Column():
            clustering_algorithm = gr.Radio(['KMeans', 'Agglomerative Hierarchical - single linkage', 'DBSCAN', 'HDBSCAN'], label='Clustering Algorithm', info='Choose the clustering algorithm.')
    gr.HTML('<h1>Choose Hyperparameters</h1>')
    with gr.Row():
        with gr.Column():
            seed = gr.Number(value=42, label='Seed (For KMeans)', info='Choose the seed for the clustering algorithm.')
            n_clusters = gr.Dropdown(['3', '5'], label='Number of clusters (For single linkage hierarchical)', info='Choose the number of clusters.', value='3')
        with gr.Column():
            with gr.Row():
                eps = gr.Number(value=0.5, label='Epsilon (For DBSCAN)', info='Choose the epsilon for DBSCAN.')
                min_samples = gr.Number(value=5, label='Min Samples (For DBSCAN)', info='Choose the min_samples for DBSCAN.')
            min_cluster_size = gr.Number(value=5, label='Min Cluster Size (For HDBSCAN)', info='Choose the min_cluster_size for HDBSCAN.')
    with gr.Row():
        metric = gr.Dropdown(['euclidean', 'manhattan', 'cosine'], label='Metric', info='Choose the metric for the clustering algorithm.', value='euclidean')
    gr.HTML('<h1>Results</h1>')
    with gr.Row():
        with gr.Column():
            silhouette = gr.Textbox(label='Silhouette Score')
        with gr.Column():
            adjusted_rand = gr.Textbox(label='Adjusted Rand Score')
    # gr.HTML('<h1>Visualization in 2D</h1>')
    # with gr.Row():
    #     true_plots = gr.ScatterPlot(x='x', y='y', value=df)
    #     cluster_plots = gr.ScatterPlot(x='x', y='y', value=pd.DataFrame)

    btn = gr.Button('Run clustering')
    btn.click(fn=run_clustering, inputs=[embedding_type, clustering_algorithm, seed, n_clusters, eps, min_samples, min_cluster_size, metric], outputs=[silhouette, adjusted_rand])#, true_plots, cluster_plots])


demo.launch()