import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load gensim word2vec model.
from gensim.models import KeyedVectors

# Load huggingface models.
from transformers import AutoTokenizer, AutoModel, BertModel, BertTokenizer
import torch



def load_data():
    """
    Load the passages from the csv file.
    :return: Reviews text and reviews ratings as lists. No filtering based on the product or the language is carried out here.
    NOTE: The title of the review is the prepended to the review text.
    """
    df = pd.read_csv("data/review_data.csv")

    # Extract the reviews.
    reviews = df["reviews.text"].tolist()
    ratings = df["reviews.rating"].tolist()

    # Concatenate the reviews.text and reviews.title.
    reviews = [str(df["reviews.title"][i]) + " " + reviews[i] for i in range(len(reviews))]

    return reviews, ratings



def store_embeddings(reviews, model_name="bert", store_path: str = ""):
    embeddings = []
    tokenizer = None
    model = None
    if model_name == 'bert':
        model = BertModel.from_pretrained('bert-base-uncased')
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    elif model_name == 't5':
        model = AutoModel.from_pretrained("t5-base")
        tokenizer = AutoTokenizer.from_pretrained("t5-base")
    elif model_name == 'word2vec':
        """
        Get the word2vec embeddings.
        """
        model = KeyedVectors.load_word2vec_format("/Users/simpleparadox/Documents/comlam_raw/GoogleNews-vectors-negative300.bin.gz", binary=True)
        for i, review in enumerate(reviews):
            # Split the review into words.
            print(f"Review {i}")
            review = review[:512]
            words = review.split()
            word_embeddings =[]
            for word in words:
                try:
                    embed = model[word]
                    word_embeddings.append(embed)
                except:
                    continue
            word_embeddings = np.mean(word_embeddings, axis=0)
            embeddings.append(word_embeddings)
        embeddings = np.array(embeddings)
        np.savez_compressed(store_path, embeddings)
        return
    else:
        raise ValueError("Invalid model name.")

    # For huggingface models.
    assert model is not None
    assert tokenizer is not None
    for i, review in enumerate(reviews):
        print("Review: ", i)
        # Truncate the review to the first 512 tokens.
        review = review[:512]
        # Tokenize the review.
        tokens = tokenizer(review, return_tensors='pt')
        # Get the embeddings.
        with torch.no_grad():
            # pooler_output = model(**tokens)['pooler_output']
            last_hidden_state = model(**tokens)['last_hidden_state']
        # Get the mean of the embeddings.
        embeddings.append(last_hidden_state.mean(axis=1).numpy()[0])
        # embeddings.append(pooler_output.numpy()[0])

    embeddings = np.array(embeddings)
    np.savez_compressed(store_path, embeddings)

