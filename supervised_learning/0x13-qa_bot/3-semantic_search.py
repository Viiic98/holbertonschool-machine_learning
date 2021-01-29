#!/usr/bin/env python3
""" Semantic Search """
import os
import numpy as np
import tensorflow_hub as hub


def semantic_search(corpus_path, sentence):
    """  performs semantic search on a corpus of documents

        - corpus_path is the path to the corpus of reference documents
          on which to perform semantic search
        - sentence is the sentence from which to perform semantic search
        Returns: the reference text of the document most similar to
                 sentence
    """
    # Load embedding model from tf-hub
    embed = hub.load("https://tfhub.dev/google/"
                     "universal-sentence-encoder-large/5")
    # Read every file and put it content into a list
    references = [sentence]
    for file in os.listdir("./{}".format(corpus_path)):
        if file.endswith(".md"):
            with open('./{}/{}'.format(corpus_path, file)) as f:
                references.append(f.read())
    # Apply embedding to the content of the files
    embeddings = embed(references)
    # Create a correlation matrix
    correlation = np.inner(embeddings, embeddings)
    # Take the best between the sentence and all the references
    closest = np.argmax(correlation[0, 1:])
    return references[closest + 1]
