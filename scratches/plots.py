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