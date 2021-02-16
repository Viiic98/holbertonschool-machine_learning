# Natural Language Processing - Word Embeddings

![n](https://miro.medium.com/max/1838/1*sAJdxEsDjsPMioHyzlN3_A.png)

## Tasks

### [Bag Of Words](./0-bag_of_words.py)
- Write a function def bag_of_words(sentences, vocab=None): that creates a bag of words embedding matrix

### [TF-IDF](./1-tf_idf.py)
- Write a function def tf_idf(sentences, vocab=None): that creates a TF-IDF embedding

### [Train Word2Vec](./2-word2vec.py)
- Write a function def word2vec_model(sentences, size=100, min_count=5, window=5, negative=5, cbow=True, iterations=5, seed=0, workers=1): that creates and trains a gensim word2vec model

### [Extract Word2Vec](./3-gensim_to_keras.py)
- Write a function def gensim_to_keras(model): that converts a gensim word2vec model to a keras Embedding layer

### [FastText](./4-fasttext.py)
- Write a function def fasttext_model(sentences, size=100, min_count=5, negative=5, window=5, cbow=True, iterations=5, seed=0, workers=1): that creates and trains a genism fastText model

### [ELMo](./5-elmo)
- When training an ELMo embedding model, you are training:

    - The internal weights of the BiLSTM
    - The character embedding layer
    - The weights applied to the hidden states
