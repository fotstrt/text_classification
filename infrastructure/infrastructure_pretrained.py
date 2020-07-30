import numpy as np
import pickle


def main():

    #load numpy arrays
    print("loading embeddings matrix")
    with open('./interm_data/pretrained_embeddings_200_200.npy', 'rb') as f:
        z = np.load(f)

    #load vocbulary
    print("loading vocabulary")
    with open('./interm_data/vocab.pkl', 'rb') as f:
        vocabulary = pickle.load(f)

    # compute tweet embedding
    counter = 1
    for fn in ['./data/train_pos_full.txt', './data/train_neg_full.txt', './data/test_data.txt']:
        print("computing embedding in file {}".format(fn))
        with open(fn, encoding='utf-8') as f:
            lines = f.readlines()
            total = len(lines)
            tweet_embeddings = np.zeros((total, z.shape[1]))
            for i, line in enumerate(lines):
                tokens = [vocabulary.get(t, -1) for t in line.strip().split()]
                tokens = [t for t in tokens if t >= 0]
                for t in tokens:
                    tweet_embeddings[i, :] += z[t, :]
                if (len(tokens) != 0):
                    tweet_embeddings[i] = tweet_embeddings[i] / len(tokens)

                if counter % 10000 == 0:
                    print(counter)
                counter += 1
        fi_1 = fn[:-4] + '_pretrained_embeddings_200_200'
        #f_2 = fi_1 +'.csv'
        np.save(fi_1, tweet_embeddings)
        #np.savetxt(f_2, tweet_embeddings, delimiter=',')

if __name__=='__main__':
    main()
