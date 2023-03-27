import gradio as gr
import numpy as np







def run_clustering(embedding_type, clustering_algorithm):
    # Write a bunch of if else conditions

    r1 = np.random.randint(0, 1000, size=1)
    return r1, r1 + 1  # The returned value is rendered as a string.



with gr.Blocks(theme=gr.themes.Soft()) as demo:
    with gr.Row():
        with gr.Column():
            embedding_type = gr.Radio(['Word2Vec - Average', 'BERT - Average', 'BERT - CLS'], label='Embedding type', info='Choose the embedding type.')
            clustering_algorithm = gr.Radio(['KMeans', 'Agglomerative Hierarchical', 'DBSCAN', 'OPTICS'], label='Clustering Algorithm',
                            info='Choose the clustering algorithm.')
        with gr.Column():
            silhouette_score = gr.Textbox(label='Silhouette Score')
            adjusted_rand_score = gr.Textbox(label='Adjusted Rand Score')

    btn = gr.Button('Run clustering')
    btn.click(fn=run_clustering, inputs=[embedding_type, clustering_algorithm], outputs=[silhouette_score, adjusted_rand_score])


demo.launch()