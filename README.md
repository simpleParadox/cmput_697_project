# CMPUT 697 Project
In this project, we apply different clustering algorithms to an Amazon review dataset.

The dataset is available at [here](https://www.kaggle.com/datasets/yasserh/amazon-product-reviews-dataset).

## Goal
The goal of this project is to apply various clustering algorithms to different embedding types and compare their performance.

What motivated this project?
Customer reviews are important for businesses. They help businesses understand their customers and improve their products. However, there are instances, 
where the ratings of the amazon reviews do not reflect the actual sentiment of the review text. For example, a review with a rating of 2 may actually have a review text that might be better suited for a 4 star rating. Using a clustering algorithm trained on text embeddings, we can accurately assign the ratings of the reviews.
In this project, we choose to look at a question that is a prerequisite to this problem. We ask the question, "What types of embeddings can we use to best represent the review text, and how does the choice of embedding affect the performance of various clustering algorithms?"


## Install python packages.
The project uses anaconda environments.
- conda create -n <env_name> python=3.9  # Where <env_name> is the name of your environment.
- conda activate <env_name>

To run the experiments, you need to have the following packages installed (you can use pip for these):
- scikit-learn
- torch
- seaborn
- pandas
- gensim
- matplotlib
- gradio
- transformers

After installing the above libraries, the hdbscan library also needs to be installed.

For mac users (you can use pip even inside the anaconda environment)
- pip install hdbscan

For windows users (use conda package manager)
- conda install -c conda-forge hdbscan

Please install these packages according to your operating system and 
package manager.

This project was built using anaconda so use a conda environment to run the experiments install the packages and run the experiments.



NOTE (for mac users only): You can install all packages directly from the 'requirements.txt' file by using pip.
```
pip install -r requirements.txt
```


## Demo
To run the demo on your local machine, run the following command:
```
python gradio_app.py
```


## To run the code.
```
python main.py -h # Gives you the list of command line arguments.
```
```
usage: main.py [-h] --embedding_type EMBEDDING_TYPE --clustering_algorithm_type CLUSTERING_ALGORITHM_TYPE [--seed SEED] [--n_clusters N_CLUSTERS] [--eps EPS] [--min_cluster_size MIN_CLUSTER_SIZE] [--metric METRIC]
               [--n_ratings N_RATINGS]

Run clustering algorithms on the Amazon reviews dataset.

optional arguments:
  -h, --help            show this help message and exit
  --embedding_type EMBEDDING_TYPE
                        The type of embedding to use.
  --clustering_algorithm_type CLUSTERING_ALGORITHM_TYPE, -alg_type CLUSTERING_ALGORITHM_TYPE
                        The clustering algorithm to use.
  --seed SEED           The random seed to use.
  --n_clusters N_CLUSTERS
                        The number of clusters to use.
  --eps EPS             The eps parameter for DBSCAN.
  --min_cluster_size MIN_CLUSTER_SIZE
                        The min_cluster_size parameter for HDBSCAN.
  --metric METRIC       The metric to use for the clustering algorithms.
  --n_ratings N_RATINGS
                        The number of ratings to use.
```

The following example command runs the density based algorithms on the 'bert_avg' embeddings.
```
python main.py --embedding_type "bert_avg" --clustering_algorithm_type density
```

## Framework

## Embeddings

NOTE: We mean center the embeddings before clustering using sklearn's StandardScaler.
## Clustering Algorithms
TODO
## Results
TODO

DBSCAN internal: eps=40.0=> 0.29167095  50.0=0.45513833, 60.0=0.4692
HDBSCAN internal = -0.0124097 for everything, even for eps=60.0

DBSCAN external: eps=40.0=>5.6732e-03  50.0=2.3222e-03, 60=0.003375
HDBSCAN external: 006082314232855933 for all eps.


## Conclusion
TODO

