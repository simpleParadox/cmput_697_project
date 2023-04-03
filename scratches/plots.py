from preprocess import load_data, store_embeddings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import seaborn as sns



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
clusters = 5
algorithm = 'hdbscan'
score_type ='internal'
if score_type == 'internal':
    internal_bert_embeddings = np.load(f"results/{algorithm}_internal_bert_embeddings_{clusters}.npz", allow_pickle=True)['arr_0'].reshape(-1).tolist()[0][f'{algorithm}_internal'][0]
    internal_bert_avg = np.load(f"results/{algorithm}_internal_bert_avg_{clusters}.npz", allow_pickle=True)['arr_0'].reshape(-1).tolist()[0][f'{algorithm}_internal'][0]
    internal_w2v_avg = np.load(f"results/{algorithm}_internal_w2v_embeddings_{clusters}.npz", allow_pickle=True)['arr_0'].reshape(-1).tolist()[0][f'{algorithm}_internal'][0]

# Load the external scores.
if score_type == 'external':
    external_bert_embeddings = np.load(f"results/{algorithm}_external_bert_embeddings_{clusters}.npz", allow_pickle=True)['arr_0'].reshape(-1)
    external_bert_avg = np.load(f"results/{algorithm}_external_bert_avg_{clusters}.npz", allow_pickle=True)['arr_0'].reshape(-1)
    external_w2v_avg = np.load(f"results/{algorithm}_external_w2v_embeddings_{clusters}.npz", allow_pickle=True)['arr_0'].reshape(-1)

# Create dataframes for seaborn.
# possible_eps = [0.5, 1.0, 5.0, 10.0, 15.0]  # For DBSCAN
possible_eps = [2, 5, 10, 15, 20]  # For HDBSCAN

# Internal
if score_type == 'internal':
    df = pd.DataFrame({'eps':np.array(possible_eps, np.int32),
                       'bert_embeddings': internal_bert_embeddings,
                       'bert_avg': internal_bert_avg,
                       'w2v_avg': internal_w2v_avg})
else:
    df = pd.DataFrame({'eps': np.array(possible_eps, np.int32),
                       'bert_embeddings': external_bert_embeddings,
                       'bert_avg': external_bert_avg,
                       'w2v_avg': external_w2v_avg})
# External
#

# Now plot the scores in seaborn.
plt.clf()
sns.set_style("whitegrid")
# df = pd.DataFrame({'DBSCAN': dbscan_internal, 'HDBSCAN': hdbscan_internal, 'eps': possible_eps})
# sns.lineplot(x='eps', y='value', data=pd.melt(df, ['eps']), hue='Embedding')
plt.plot(possible_eps, df['bert_embeddings'], label='BERT - CLS ', c='red')
plt.plot(possible_eps, df['bert_avg'], label='BERT - Average ', c='blue')
plt.plot(possible_eps, df['w2v_avg'], label='Word2Vec - Average ', c='green')
plt.xlabel("Min Cluster Size", fontsize=14)
plt.xticks(possible_eps, fontsize=12, rotation=90)

if score_type == 'internal':
    plt.title(f"Internal Validation Scores for {algorithm}", fontsize=14)
    plt.ylabel("Silhouette Score", fontsize=14)
else:
    plt.title(f"External Validation Scores for {algorithm} for {clusters} clusters", fontsize=14)
    plt.ylabel("Adjusted Rand Score", fontsize=14)

plt.legend(fontsize=12)
plt.tight_layout()
if score_type == 'internal':
    plt.savefig(f"plots/score_plots/{algorithm}_{score_type}.pdf")
# plt.show()



