import csv
from collections import Counter
import pickle
import numpy as np
import re

pos_path = '/path_to_pos_path/train_pos_full.txt'
neg_path = '/path_to_neg_pathtrain_neg_full.txt'
glove_path = '/path_to_glove_file/glove.42B.300d.txt'
test_path = '/path_to_test_path/test_data.txt'

vocab_size = 100000
emb_size = 300

def preprocess_sentence(sentence):
    proc_sentence = sentence.strip()
    proc_sentence = proc_sentence.lower()

    # remove non ascii
    proc_sentence = ''.join([i if (ord(i) < 128) else ' ' for i in proc_sentence])

    # replace numbers with 'number'
    proc_sentence = re.sub("\d+", " number ", proc_sentence)    

    # separate each dot (.)
    proc_sentence = re.sub("\.", " . ", proc_sentence)

    # remove < >
    proc_sentence = re.sub("<", " ", proc_sentence)
    proc_sentence = re.sub(">", " ", proc_sentence)

    # break dashes
    proc_sentence = re.sub("-", " - ", proc_sentence)

    # split hashtags
    proc_sentence = re.sub("#", " # ", proc_sentence)  

    # break abbreviations
    proc_sentence = re.sub("'", " ' ", proc_sentence)

    # replace extra spaces
    proc_sentence = re.sub(" +", " ", proc_sentence)    


    proc_sentence = proc_sentence.split()

    return proc_sentence


# ---- preprocess test data --------
# ---- save to test_data_proc.txt --

with open(test_path, 'r') as f:
    lines = f.readlines()

test_sentences = list()
for line in lines:
    proc_line = preprocess_sentence(line)
    test_sentences.append(proc_line)


with open('test_data_proc.txt', 'w') as f:
    for sen in test_sentences:
        sen = ' '.join(sen)
        sen = sen[8:]
        sen = sen.strip()
        print(sen, file=f)


# ---- proprocess train data ------------
# ---- save to final_dataset.txt --------

labels = list()
tweets = list()

for label, f_path in enumerate([neg_path, pos_path]):

    with open(f_path, 'r') as f:

        sentences = f.readlines()

    for sen in sentences:

        labels.append(label)
        tweets.append(preprocess_sentence(sen))




with open('final_dataset.txt', 'w') as f:
    for label, tweet in zip(labels, tweets):
        tweet = ' '.join(tweet)
        print_line = '{} {}'.format(tweet, label)
        print(print_line, file=f)


# ----------- create word_index file ------------
# -------- and glove embeddings pickled array ---


word_list = list()
for tweet in tweets:
    for word in tweet:
        word_list.append(word)

word_freq_list = Counter(word_list).most_common(vocab_size - 2)

print(len(word_freq_list))

word_index = dict()

word_index["padid"] = 0
word_index["unkid"] = 1

for word, _ in word_freq_list:
    if word != 'unkid':
        word_index[word] = len(word_index)


embeddings_index = {}
f = open(glove_path, encoding='utf8')
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

glove_embedding_matrix = np.zeros((vocab_size, emb_size))
for word, i in word_index.items():
    if i < vocab_size:
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            glove_embedding_matrix[i] = embedding_vector
        else:
            glove_embedding_matrix[i] = embeddings_index.get('unk')

    glove_embedding_matrix[0] = np.random.normal(0, 1, emb_size)


#with open("word_index.pickle", "wb") as f:
#    pickle.dump(word_index, f)

with open('word_index.txt', 'w') as f:
    for k, v in word_index.items():
        print('{} {}'.format(k, v), file=f)
    
with open("embeddings.pickle", "wb") as f:
    pickle.dump(glove_embedding_matrix, f)