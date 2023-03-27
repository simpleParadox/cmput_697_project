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



def run_clustering(embedding_type, clustering_algorithm, seed, n_clusters, eps, min_samples, metric):
    # Write a bunch of if else conditions
    print("Running clustering")
    print("Number of clusters: ", n_clusters)
    n_clusters = int(n_clusters)

    reviews, ratings, good_indices = load_review_and_ratings()

    if n_clusters == 3:
        # Relabel the ratings if the number of selected clusters is 3, else leave them unchanged.
        ratings[ratings < 3] = 0
        ratings[ratings == 3] = 1
        ratings[ratings > 3] = 2


    embedding = None
    model = None

    if embedding_type == 'Word2Vec - Average':
        embedding = np.load(f"embeds/w2v_embeddings.npz")['arr_0']
        embedding = embedding[good_indices]

    elif embedding_type == 'BERT - Average':
        embedding = np.load(f"embeds/bert_avg.npz")['arr_0']
        embedding = embedding[good_indices]

    elif embedding_type == 'BERT - CLS':
        embedding = np.load(f"embeds/bert_embeddings.npz")['arr_0']
        embedding = embedding[good_indices]

    assert embedding is not None

    if clustering_algorithm == 'KMeans':
        model = KMeans(n_clusters=n_clusters, random_state=seed)
        model.fit(embedding)

    elif clustering_algorithm == 'Agglomerative Hierarchical':
        model = AgglomerativeClustering(n_clusters=n_clusters)
        model.fit(embedding)

    elif clustering_algorithm == 'DBSCAN':
        model = DBSCAN(eps=eps, min_samples=min_samples, metric=metric)
        model.fit(embedding)

    elif clustering_algorithm == 'OPTICS':
        model = OPTICS(eps=eps, min_samples=min_samples, metric=metric)
        model.fit(embedding)


    # Evaluate the model.
    assert model is not None
    silhouette = silhouette_score(embedding, model.labels_)
    ari = adjusted_rand_score(ratings, model.labels_)

    return silhouette, ari


with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.HTML('<h1>Choose Embedding type and Clustering Algorithm</h1>')
    with gr.Row():
        with gr.Column():
            embedding_type = gr.Radio(['Word2Vec - Average', 'BERT - Average', 'BERT - CLS'], label='Embedding type', info='Choose the embedding type.')
        with gr.Column():
            clustering_algorithm = gr.Radio(['KMeans', 'Agglomerative Hierarchical', 'DBSCAN', 'OPTICS'], label='Clustering Algorithm', info='Choose the clustering algorithm.')
    gr.HTML('<h1>Choose Hyperparameters</h1>')
    with gr.Row():
        with gr.Column():
            seed = gr.Number(value=42, label='Seed', info='Choose the seed for the clustering algorithm.')
            n_clusters = gr.Dropdown(['3', '5'], label='Number of clusters', info='Choose the number of clusters.')
        with gr.Column():
            eps = gr.Number(value=0.5, label='Epsilon', info='Choose the epsilon for Density based algorithms.')
            min_samples = gr.Number(value=5, label='Min Samples', info='Choose the min samples for Density based algorithms.',)
    with gr.Row():
        metric = gr.Dropdown(['euclidean', 'manhattan', 'cosine'], label='Metric', info='Choose the metric for the clustering algorithm.', value='cosine')
    gr.HTML('<h1>Results</h1>')
    with gr.Row():
        with gr.Column():
            silhouette = gr.Textbox(label='Silhouette Score')
        with gr.Column():
            adjusted_rand = gr.Textbox(label='Adjusted Rand Score')

    btn = gr.Button('Run clustering')
    btn.click(fn=run_clustering, inputs=[embedding_type, clustering_algorithm, seed, n_clusters, eps, min_samples, metric], outputs=[silhouette, adjusted_rand])


demo.launch()