#!/usr/bin/env python3
from scipy.sparse import *
import numpy as np
import pickle
import random
import pandas as pd


def main():
    # load numpy arrays
    print("loading embeddings matrix")
    with open('./interm_data/embeddings_200_200.npz', 'rb') as f:
        npz_file = np.load(f)
        x = npz_file['arr_0']
        y = npz_file['arr_1']
    
    # load vocbulary
    print("loading vocabulary")
    with open('./interm_data/vocab.pkl', 'rb') as f:
        vocab = pickle.load(f)

    # compute final embedding for each word
    print("computing final embedding for each word")
    z = x + y

    print(z[vocab.get('<user>', -1), :])


    #start reading from tweet file,
    #if in vocabulary, then take its embeddings
    # match word to id from vocab.pkl, then xs[]


    with open('./data/glove.twitter.pretrained/glove.twitter.27B.200d.txt', encoding='utf-8') as f:
        for line in f:
            word = line[:line.find(' ')]
            temp = [float(s) for s in line[1 + line.find(' '):-1].split()]
            val = vocab.get(word, -1)
            if (val >= 0):
                #print (word)
                #print (val)
                #print(z[val, :])
                z[val, :] = temp
                #print(z[val, :])
                #print(z[val+1, :])
	
    print(z[vocab.get('<user>', -1), :])
    '''
    eta = 0.001
    alpha = 3 / 4

    epochs = 10 #change to 10

    for epoch in range(epochs):
        print("epoch {}".format(epoch))
        for ix, jy, n in zip(cooc.row, cooc.col, cooc.data):
            logn = np.log(n)
            fn = min(1.0, (n / nmax) ** alpha)
            x, y = xs[ix, :], ys[jy, :]
            scale = 2 * eta * fn * (logn - np.dot(x, y))
            xs[ix, :] += scale * y
            ys[jy, :] += scale * x
			# fill in your SGD code here,
			# for the update resulting from co-occurence (i,j)

    '''
    np.save('./interm_data/pretrained_embeddings_200_200', z)


if __name__ == '__main__':
    main()
