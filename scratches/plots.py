from preprocess import load_data, store_embeddings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import seaborn as sns

from scipy.stats import sem


# Load the data
reviews, ratings = load_data()

# Get the indices where the ratings are not nan.
good_indices = np.where(np.isnan(ratings) == False)[0]
ratings = np.array(ratings)[good_indices]

# Make a dataframe with the ratings.
df = pd.DataFrame({'rating': ratings.astype(int)})
plt.clf()
sns.histplot(data=df, x='rating', shrink=0.8, bins=5, discrete=True, hue='rating', palette=sns.color_palette("Spectral", as_cmap=True))
plt.xlabel("Rating", fontsize=12)
plt.ylabel("Count", fontsize=12)
plt.title("Distribution of ratings", fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.savefig("plots/rating_distribution.pdf")



# Plotting the internal and external validation scores for the different number of clusters.
# First let's plot the internal validation scores for 3 clusters for the density based clustering algorithm.
# One graph will contain the scores for all the three embeddings.

# Load the scores for the density based algorithms.
# First load the internal scores.
clusters = 3
algorithm = 'agglomerative'
score_type ='results'

if score_type == 'internal':
    internal_bert_embeddings = np.load(f"results/{algorithm}_internal_bert_embeddings_{clusters}.npz", allow_pickle=True)['arr_0'].reshape(-1).tolist()[0][f'{algorithm}_internal'][0]
    internal_bert_avg = np.load(f"results/{algorithm}_internal_bert_avg_{clusters}.npz", allow_pickle=True)['arr_0'].reshape(-1).tolist()[0][f'{algorithm}_internal'][0]
    internal_w2v_avg = np.load(f"results/{algorithm}_internal_w2v_embeddings_{clusters}.npz", allow_pickle=True)['arr_0'].reshape(-1).tolist()[0][f'{algorithm}_internal'][0]

# Load the external scores.
if score_type == 'external':
    external_bert_embeddings = np.load(f"results/{algorithm}_external_bert_embeddings_{clusters}.npz", allow_pickle=True)['arr_0'].reshape(-1)
    external_bert_avg = np.load(f"results/{algorithm}_external_bert_avg_{clusters}.npz", allow_pickle=True)['arr_0'].reshape(-1)
    external_w2v_avg = np.load(f"results/{algorithm}_external_w2v_embeddings_{clusters}.npz", allow_pickle=True)['arr_0'].reshape(-1)

if score_type == 'results':
    # Load the results for Agglomerative Clustering only.
    internal_bert_embeddings = np.load(f"results/{algorithm}_internal_results_bert_embeddings.npz", allow_pickle=True)['arr_0'].reshape(-1)
    internal_bert_avg = np.load(f"results/{algorithm}_internal_results_bert_avg.npz", allow_pickle=True)['arr_0'].reshape(-1)
    internal_w2v_avg = np.load(f"results/{algorithm}_internal_results_w2v_embeddings.npz", allow_pickle=True)['arr_0'].reshape(-1)

if score_type == 'both':
    # For KMeans only.
    internal_bert_embeddings = np.load(f"results/{algorithm}_results_bert_embeddings_{clusters}_50_iters_fixed_seed.npz", allow_pickle=True)['arr_0'].tolist()[f'{algorithm}_internal']
    internal_bert_avg = np.load(f"results/{algorithm}_results_bert_avg_{clusters}_50_iters_fixed_seed.npz", allow_pickle=True)['arr_0'].tolist()[f'{algorithm}_internal']
    internal_w2v_avg = np.load(f"results/{algorithm}_results_w2v_embeddings_{clusters}_50_iters_fixed_seed.npz", allow_pickle=True)['arr_0'].tolist()[f'{algorithm}_internal']

    external_bert_embeddings = np.load(f"results/{algorithm}_results_bert_embeddings_{clusters}_50_iters_fixed_seed.npz", allow_pickle=True)['arr_0'].tolist()[f'{algorithm}_external']
    external_bert_avg = np.load(f"results/{algorithm}_results_bert_avg_{clusters}_50_iters_fixed_seed.npz", allow_pickle=True)['arr_0'].tolist()[f'{algorithm}_external']
    external_w2v_avg = np.load(f"results/{algorithm}_results_w2v_embeddings_{clusters}_50_iters_fixed_seed.npz", allow_pickle=True)['arr_0'].tolist()[f'{algorithm}_external']

# Create dataframes for seaborn.
# possible_eps = [0.5, 1.0, 5.0, 10.0, 15.0]  # For DBSCAN
possible_eps = [2, 5, 10, 15, 20]  # For HDBSCAN
possible_clusters = [n_cluster for n_cluster in range(3, 20, 2)]

# Internal
if score_type == 'internal':
    df = pd.DataFrame({'eps':np.array(possible_eps, np.int32),
                       'bert_embeddings': internal_bert_embeddings,
                       'bert_avg': internal_bert_avg,
                       'w2v_avg': internal_w2v_avg})
elif score_type == 'external':
    df = pd.DataFrame({'eps': np.array(possible_eps, np.int32),
                       'bert_embeddings': external_bert_embeddings,
                       'bert_avg': external_bert_avg,
                       'w2v_avg': external_w2v_avg})
elif score_type == 'results':
    df = pd.DataFrame({'n_clusters': np.array(possible_clusters),
                       'bert_embeddings': internal_bert_embeddings,
                       'bert_avg': internal_bert_avg,
                       'w2v_avg': internal_w2v_avg})
# For KMeans.
elif score_type == 'both':
    df = pd.DataFrame({'n_clusters': np.array(possible_clusters),
                       'bert_embeddings': np.mean(internal_bert_embeddings, axis=0),
                       'bert_avg': np.mean(internal_bert_avg, axis=0),
                       'w2v_avg': np.mean(internal_w2v_avg, axis=0),
                       'bert_embeddings_sem': sem(internal_bert_embeddings, axis=0),
                       'bert_avg_sem': sem(internal_bert_avg, axis=0),
                        'w2v_avg_sem': sem(internal_w2v_avg, axis=0)})
    df_external = pd.DataFrame({'n_clusters': np.array(possible_clusters),
                       'bert_embeddings': np.mean(external_bert_embeddings, axis=0),
                       'bert_avg': np.mean(external_bert_avg, axis=0),
                       'w2v_avg': np.mean(external_w2v_avg, axis=0),
                       'bert_embeddings_sem': sem(external_bert_embeddings, axis=0),
                       'bert_avg_sem': sem(external_bert_avg, axis=0),
                        'w2v_avg_sem': sem(external_w2v_avg, axis=0)})
#

# Now plot the scores in seaborn.
plt.clf()
sns.set_style("whitegrid")
# df = pd.DataFrame({'DBSCAN': dbscan_internal, 'HDBSCAN': hdbscan_internal, 'eps': possible_eps})
# sns.lineplot(x='eps', y='value', data=pd.melt(df, ['eps']), hue='Embedding')
plt.plot(possible_clusters, df['bert_embeddings'], label='BERT - CLS ', c='red')
plt.plot(possible_clusters, df['bert_avg'], label='BERT - Average ', c='blue')
plt.plot(possible_clusters, df['w2v_avg'], label='Word2Vec - Average ', c='green')
plt.xlabel("Number of clusters", fontsize=14)
plt.xticks(possible_clusters, fontsize=12, rotation=90)

if score_type == 'internal' or score_type == 'both':
    plt.title(f"Internal Validation Scores for {algorithm}", fontsize=14)
    plt.ylabel("Silhouette Score", fontsize=14)
    if score_type == 'both':
        plt.fill_between(possible_clusters, df['bert_embeddings'] - df['bert_embeddings_sem'], df['bert_embeddings'] + df['bert_embeddings_sem'], alpha=0.2, color='red')
        plt.fill_between(possible_clusters, df['bert_avg'] - df['bert_avg_sem'], df['bert_avg'] + df['bert_avg_sem'], alpha=0.2, color='blue')
        plt.fill_between(possible_clusters, df['w2v_avg'] - df['w2v_avg_sem'], df['w2v_avg'] + df['w2v_avg_sem'], alpha=0.2, color='green')
elif score_type == 'external':
    plt.title(f"External Validation Scores for {algorithm} for {clusters} clusters", fontsize=14)
    plt.ylabel("Adjusted Rand Score", fontsize=14)
if score_type == 'results':
    plt.title(f"Internal Validation Scores for {algorithm} single linkage", fontsize=14)
    plt.ylabel("Silhouette Score", fontsize=14)


plt.legend(fontsize=12)
plt.tight_layout()
plt.savefig(f"plots/score_plots/{algorithm}_internal_validation.pdf")
# plt.show()



