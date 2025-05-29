import gensim.downloader as api
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Load GloVe embeddings (Twitter, 50 dimensions)
wv = api.load('glove-twitter-50')

# Given words
words = ['dog', 'bark', 'tree', 'bank', 'river', 'money']

# Get word vectors (skip words not in vocabulary)
word_vectors = []
valid_words = []
for word in words:
    if word in wv:
        word_vectors.append(wv[word])
        valid_words.append(word)
    else:
        print(f"'{word}' not found in vocabulary.")

# Compute cosine similarity matrix
cos_sim_matrix = cosine_similarity(word_vectors)

# Print results
print("Words used:", valid_words)
print("\nCosine Similarity Matrix (GloVe-Twitter-50D):")
print(np.round(cos_sim_matrix, 2))
