import numpy as np
import pickle


def main():

    #load numpy arrays
    print("loading embeddings matrix")
    with open('./interm_data/embeddings.npz', 'rb') as f:
        npz_file = np.load(f)
        x = npz_file['arr_0']
        y = npz_file['arr_1']

    #load vocbulary
    print("loading vocabulary")
    with open('./interm_data/vocab.pkl', 'rb') as f:
        vocabulary = pickle.load(f)

    #compute final embedding for each word
    print("computing final embedding for each word")
    z = x + y

    # compute tweet embedding
    counter = 1
    for fn in ['./data/train_pos_full.txt', './data/train_neg_full.txt', './data/test_data.txt']:
        print("computing embedding in file {}".format(fn))
        with open(fn, encoding='utf-8') as f:
            lines = f.readlines()
            total = len(lines)
            tweet_embeddings = np.zeros((total, y.shape[1]))
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
        fi_1 = fn[:-4] + '_embeddings'
        f_2 = fi_1 +'.csv'
        np.save(fi_1, tweet_embeddings)
        np.savetxt(f_2, tweet_embeddings, delimiter=',')

if __name__=='__main__':
    main()