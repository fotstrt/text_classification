#!/usr/bin/env python3
import pickle

# Gets the file "vocab_cut.txt" and creates a dictionary, where {key} is a token and {value} is its index in "vocab_cut.txt"

def main():
    vocab = dict()
    with open('vocab_cut.txt') as f:
        i=0
        for idx, line in enumerate(f):
            vocab[line.strip()] = idx

    with open('vocab.pkl', 'wb') as f:
        pickle.dump(vocab, f, pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    main()
