import xgboost 
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

import utils
import numpy as np
import pandas as pd

def process_tweets():
    print("loading training embeddings matrices")

    with open('./final_data/train_pos_full_preprocessed_1_no_dupes_pretrained_embeddings_50.npy', 'rb') as f:
        x_pos = np.load(f)
        y_pos = np.ones(x_pos.shape[0])

    with open('./final_data/train_neg_full_preprocessed_1_no_dupes_pretrained_embeddings_50.npy', 'rb') as f:
        x_neg = np.load(f)
        y_neg = np.zeros(x_neg.shape[0])

    with open('./final_data/test_data_preprocessed_1_pretrained_embeddings_50.npy', 'rb') as f:
        x_test = np.load(f)

    #concatenate positive, negative samples
    x = np.vstack((x_pos, x_neg))
    y = np.hstack((y_pos, y_neg))

    #shuffle the data
    s = np.arange(x.shape[0])
    np.random.shuffle(s)

    return x[s],y[s],x_test

if __name__ == '__main__':

    #np.random.seed(1337)

    #load_tweet_embeddings
    X_train, Y_train, X_test = process_tweets()

    #Normalization
    #scaler = StandardScaler().fit(X_train)
    #X_train_scaled = scaler.transform(X_train)
    #X_test_scaled = scaler.transform(X_test)

    # Cross Validation process
    print('model selection using cross-validation')
    model = xgboost.XGBClassifier(nthread = -1)
    #parameters = {'kernel': ['rbf', 'sigmoid', 'linear'], 'C': [0.1, 1, 10, 20, 50]}
    parameters = {'max_depth':[30], 'eta':[0.2], 'n_estimators':[50]}
    grid = GridSearchCV(model, parameters, cv=5, scoring='accuracy', verbose=10, n_jobs=-1)
    grid.fit(X_train, Y_train)
    print('HERE')
    results = pd.DataFrame.from_dict(grid.cv_results_).to_csv('./implementations/results/results_2_pretrained/xgboost_grid_results_50.csv', index=True)

    #Final training
    print(grid.best_params_)
    print(grid.best_score_)
    #print('optimal classifier training')
    #clf = svm.LinearSVC(C=0.1)
    #clf.fit(X_train, Y_train)

    #Testing
    print('optimal classifier testing')
    #predictions = clf.predict(X_test)
    predictions = grid.best_estimator_.predict(X_test)
    predictions[predictions == 0] = -1

    #produce output file
    print('producing output file')
    predictions_final = [(str(j+1), int(predictions[j])) for j in range(predictions.shape[0])]
    utils.save_results_to_csv(predictions_final, './implementations/results/results_2_pretrained/xgboost_50.csv')
    print ('\nSaved to xgboost_50.csv')
