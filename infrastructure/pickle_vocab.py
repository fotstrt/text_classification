#!/usr/bin/env python3
import pickle


def main():
    vocab = dict()
    with open('./interm_data/vocab_cut_preprocessed_1_no_dupes.txt') as f:
        for idx, line in enumerate(f):
            vocab[line.strip()] = idx

    print(len(vocab.keys()))

    with open('./interm_data/vocab_preprocessed_1_no_dupes.pkl', 'wb') as f:
        pickle.dump(vocab, f, pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    main()
