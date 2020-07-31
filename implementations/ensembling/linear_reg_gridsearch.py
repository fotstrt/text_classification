# Importing and installing necessary libaries
import numpy as np
import pandas as pd
import ast
import os
import re
import sys
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import nltk
from nltk.corpus import RegexpTokenizer as regextoken
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from xgboost import XGBClassifier
import gensim
from gensim.models.keyedvectors import KeyedVectors
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report
from datetime import datetime
#!pip install bert-tensorflow

from sklearn.feature_selection import SelectFromModel


import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, accuracy_score



data = pd.read_csv('../../csv_data/train_data_full-not-processed.csv', header=None,  index_col=0)
data.columns=["Label", "Sentence"]
data = data.dropna()

_, val_set = train_test_split(data, test_size=0.1, random_state=42)

y_train=list(val_set['Label'])
X = []
X_test = []

for myfile in ['X_val_bert_hg.csv', 'X_val_roberta_base.csv', 'X_val_lstm.csv', 'X_val_roberta.csv', 'X_val_roberta_lstm.csv', 'X_val_bert_retrained.csv', 'X_val_bert_large.csv']:  
    X_pd = pd.read_csv('predictions_new/' + myfile)

    X_tmp = [ast.literal_eval(x) for x in X_pd['X']]

    X.append(X_tmp) 
    
    
    found_labels = list(np.argmax(X_tmp, axis=1))
    print('model:', myfile, 'accuracy_score:', accuracy_score(found_labels, y_train))


from sklearn.linear_model import LinearRegression

print('linear reg started...')

print('x_train', x_train[:10])
print('y_train', y_train[:10])

x_array = np.asarray(x_train)
x_array = x_array[:, 0::2]


y_array = np.asarray(y_train)
y_array = 1 - y_array

reg = LinearRegression().fit(x_array, y_array)
print(reg.score(x_array, y_array),reg.coef_)

print(type(reg.coef_))

pred_train_y = np.sum([x_array[:,i] * reg.coef_[i] for i in range(int(len(x_train[0])/2))], axis=0)

print('reg.coef_:', reg.coef_)
print('x_array[:10]:', x_array[:10])
print('y_array[:10]:', y_array[:10])
print('pred_train_y[:10]:', pred_train_y[:10])
print('len(pred_train_y):', len(pred_train_y))


thresholds = []
for thresh in np.arange(0.1, 0.801, 0.01):
    thresh = np.round(thresh, 2)
    res = accuracy_score(y_train, (pred_train_y < thresh).astype(int))
    thresholds.append([thresh, res])
    print("accuracy score at threshold {0} is {1}".format(thresh, res))

thresholds.sort(key=lambda x: x[1], reverse=True)
best_thresh = thresholds[0][0]
print("Best threshold: ", best_thresh)

'''

X_test = []

for myfile in ['X_test_bert_hg.csv', 'X_test_roberta_base.csv', 'X_test_lstm.csv', 'X_test_roberta.csv', 'X_test_roberta_lstm.csv', 'X_test_bert_retrained.csv', 'X_test_bert_large.csv']:
    X_pd = pd.read_csv('predictions_new/' + myfile)
    X_tmp = [ast.literal_eval(x) for x in X_pd['X']]
    X_test.append(X_tmp)

x_test = np.hstack(X_test)
x_test_array = np.asarray(x_test)
x_test_array = x_test_array[:, 0::2]

best_thresh = 0.48
coef_ =  [0.08289299, 0.04246045, 0.04646634, 0.28295564, 0.31884895, 0.1288666, 0.08480486]


pred_test_y = np.sum([x_test_array[:,i] * coef_[i] for i in range(int(len(x_test[0])/2))], axis=0)
best_test_pred = (pred_test_y < best_thresh).astype(int)

idx_list = range(1, len(best_test_pred) + 1)
df = pd.DataFrame(list(zip(idx_list, best_test_pred)), columns =['Id', 'Prediction'])
df = df.replace(0, -1)
print(df)

out_file = "linear_regression_ensemble_new.csv"
print('writing predictions for test set in', out_file)
df.to_csv(out_file, index=False)
'''
