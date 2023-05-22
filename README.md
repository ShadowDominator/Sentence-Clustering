# Sentence-Clustering
This is a code repository for a Sentence Clustering application. The application uses sentence embeddings and K-means clustering to group sentences based on their similarities.
# Sentence Clustering

This is a code repository for a Sentence Clustering application. The application uses sentence embeddings and K-means clustering to group sentences based on their similarities.

## Installation
Clone the repository:
```shell
git clone https://github.com/ShadowDominator/Sentence-Clustering.git
```
Install the required packages: 
```shell
pip install gradio sentence_transformers scikit-learn pandas
```
## Usage
1. Import the necessary libraries:
``` shell
import gradio as gr
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
import pandas as pd
```
2. Load the pre-trained sentence embedding model:
``` shell
embedder = SentenceTransformer('all-MiniLM-L6-v2')
```
3. Define an example set of sentences:
``` shell 
example = {'sentence': [
  "Today is a beautiful day, with clear blue skies and a gentle breeze.",
  "I love to read books and explore new ideas and concepts.",
  "My favorite hobby is hiking in the mountains and enjoying the stunning views.",
  "I am grateful for my family and friends, who always support and encourage me.",
  "Life is full of challenges and opportunities, and it's up to us to make the most of them.",
  "The sound of the waves crashing on the shore is incredibly soothing to me.",
  "I believe that laughter is the best medicine for any problem or difficulty in life.",
  "Learning a new language is a challenging but rewarding experience.",
  "The beauty of nature always fills me with a sense of awe and wonder.",
  "I am constantly amazed by the resilience and strength of the human spirit in the face of adversity."
]}
df_example = pd.DataFrame(example)
```
4. Define the sentence clustering function:
``` shell
def sentence(k_value, all_sentence):
    length = all_sentence['sentence'].apply(lambda x: len(x))
    all_sentence['class'] = length
    corpus = [i for i in all_sentence['sentence']]
    corpus_embeddings = embedder.encode(corpus)
    # Perform k-means clustering
    num_clusters = int(k_value)
    clustering_model = KMeans(n_clusters=num_clusters)
    clustering_model.fit(corpus_embeddings)
    cluster_assignment = clustering_model.labels_

    clustered_sentences = [[] for i in range(num_clusters)]
    for sentence_id, cluster_id in enumerate(cluster_assignment):
        clustered_sentences[cluster_id].append(corpus[sentence_id])
    df = pd.DataFrame(columns=['class', 'sentence'])

    for i, cluster in enumerate(clustered_sentences):
        for sentence in cluster:
            df = pd.concat([df, pd.DataFrame({'class': chr(65 + i), 'sentence': sentence}, index=[0])], ignore_index=True)

    return df
```

5. Create the Gradio interface:
``` shell
with gr.Blocks(title="Sentence Clustering") as demo:
    with gr.Row():
        with gr.Column():
            with gr.Row():
                with gr.Column(min_width=20):
                    num = gr.Number(label="Number of clustering", value=4)
                with gr.Column():
                    pass

            inputs = [
                num,
                gr.Dataframe(
                    value=df_example,
                    datatype=["str"],
                    col_count=(1, False)
                ),
            ]
        with gr.Column():
            outputs = gr.Dataframe(
                headers=["class", "sentence"],
                datatype=["str", "str"],
            )

    greet_btn = gr.Button("RUN")
    greet_btn.click(fn=sentence, inputs=inputs, outputs=outputs, api_name="Sentence Clustering")

demo.launch()
```



