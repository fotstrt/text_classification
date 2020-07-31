import csv
from collections import Counter
import pickle
import numpy as np
import re
import pandas as pd

glove_path = 'glove_tweets/glove.twitter.27B.200d.txt'
test_path = '/path_to_test_path/test_data.txt'

TRAIN_FILE_FULL = '../../csv_data/train_data_full-punct.csv'

vocab_size = 500000
emb_size = 300


labels = list()
tweets = list()

data = pd.read_csv(TRAIN_FILE_FULL, header=None,  index_col=0)
data.columns=["Label", "Sentence"]

data = data.dropna()

labels = data['Label'].tolist()
tweets = data['Sentence'].tolist()


# ----------- create word_index file ------------
# -------- and glove embeddings pickled array ---


word_list = list()
for tweet in tweets:
    tweet_list = tweet.split()
    #print(tweet_list)
    for word in tweet_list:
        word_list.append(word)

word_freq_list = Counter(word_list).most_common(vocab_size - 2)

print(len(word_freq_list))

word_index = dict()

word_index["padid"] = 0
word_index["unkid"] = 1

for word, _ in word_freq_list:
    if word != 'unkid':
        word_index[word] = len(word_index)


print("1) create embeddings index from", glove_path)

embeddings_index = {}
f = open(glove_path, encoding='utf8')
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()



print("2) create embeddings matrix")

glove_embedding_matrix = np.zeros((vocab_size, emb_size))


for word, i in word_index.items():
    if i < vocab_size:
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            glove_embedding_matrix[i] = embedding_vector
        else:
            glove_embedding_matrix[i] = embeddings_index.get('unk')

    glove_embedding_matrix[0] = np.random.normal(0, 1, emb_size)


print("3) save")

with open('word_index.txt', 'w') as f:
    for k, v in word_index.items():
        print('{} {}'.format(k, v), file=f)
    
with open("embeddings.pickle", "wb") as f:
    pickle.dump(glove_embedding_matrix, f)
