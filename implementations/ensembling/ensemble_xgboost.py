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
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import gensim
from gensim.models.keyedvectors import KeyedVectors
from keras.models import Sequential, Model
from keras.layers import Dense, Embedding, LSTM, Dropout
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from keras.regularizers import l1, l2
from sklearn.metrics import classification_report
import tensorflow as tf
import tensorflow_estimator
import tensorflow_hub as hub
from datetime import datetime
#!pip install bert-tensorflow

from sklearn.feature_selection import SelectFromModel

import tensorflow as tf
print(tf.__version__)

import bert
from bert import run_classifier
from bert import optimization
from bert import tokenization
import warnings
# warnings.filterwarnings("ignore")

import logging
import math

import pickle

import xgboost as xgb
from xgboost import XGBClassifier

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, accuracy_score

print("starting")

data = pd.read_csv('../../csv_data/train_data_full-not-processed.csv',
                   index_col = 0,
                   names=["id", "sentiment", "tweet"])
data = data.dropna()

print("The number of rows and columns in the data is: {}".format(data.shape))

# Checking the target class balance
print(data["sentiment"].value_counts())


# Preparing training and test data for BERT
label_encoder = preprocessing.LabelEncoder()
data['sentiment'] = label_encoder.fit_transform(data['sentiment'])

trainingSet, valSet = train_test_split(data, test_size=0.1, random_state=42)

# Defining input and label columns
DATA_COLUMN = 'tweet'
LABEL_COLUMN = 'sentiment'
# label_list is the list of labels, i.e. True, False or 0, 1 or 'dog', 'cat'
label_list = [0, 1]

# This is a path to an uncased (all lowercase) version of BERT
BERT_MODEL_HUB = "/cluster/scratch/atriantaf/twitter-sentiment-analysis/bert-model-hub-large/"

def create_tokenizer_from_hub_module():
    """Get the vocab file and casing info from the Hub module."""
    with tf.Graph().as_default():
        bert_module = hub.Module(BERT_MODEL_HUB)
        tokenization_info = bert_module(signature="tokenization_info", as_dict=True)
        with tf.Session() as sess:
            vocab_file, do_lower_case = sess.run([tokenization_info["vocab_file"],
                                                  tokenization_info["do_lower_case"]])

    return bert.tokenization.FullTokenizer(
        vocab_file=vocab_file, do_lower_case=do_lower_case)

tokenizer = create_tokenizer_from_hub_module()

# We'll set sequences to be at most 100 tokens long
MAX_SEQ_LENGTH = 50
def create_model(is_predicting, input_ids, input_mask, segment_ids, labels, num_labels):
    """Creates a classification model."""

    bert_module = hub.Module(BERT_MODEL_HUB, trainable=True)
    bert_inputs = dict(input_ids=input_ids, input_mask=input_mask, segment_ids=segment_ids)
    bert_outputs = bert_module(inputs=bert_inputs, signature="tokens", as_dict=True)

    # Use "pooled_output" for classification tasks on an entire sentence.
    # Use "sequence_outputs" for token-level output.
    output_layer = bert_outputs["pooled_output"]

    hidden_size = output_layer.shape[-1].value

    # Create our own layer to tune for politeness data.
    output_weights = tf.get_variable("output_weights", [num_labels, hidden_size], initializer=tf.truncated_normal_initializer(stddev=0.02))

    output_bias = tf.get_variable("output_bias", [num_labels], initializer=tf.zeros_initializer())

    with tf.variable_scope("loss"):

        # Dropout helps prevent overfitting
        output_layer = tf.nn.dropout(output_layer, keep_prob=0.9)

        logits = tf.matmul(output_layer, output_weights, transpose_b=True)
        logits = tf.nn.bias_add(logits, output_bias)
        log_probs = tf.nn.log_softmax(logits, axis=-1)

        # Convert labels into one-hot encoding
        one_hot_labels = tf.one_hot(labels, depth=num_labels, dtype=tf.float32)

        predicted_labels = tf.squeeze(tf.argmax(log_probs, axis=-1, output_type=tf.int32))
        # If we're predicting, we want predicted labels and the probabiltiies.
        if is_predicting:
            return (predicted_labels, log_probs)

        # If we're train/eval, compute loss between predicted and actual label
        per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
        loss = tf.reduce_mean(per_example_loss)
        return (loss, predicted_labels, log_probs)

# model_fn_builder actually creates our model function
# using the passed parameters for num_labels, learning_rate, etc.
def model_fn_builder(num_labels, learning_rate, num_train_steps,
                     num_warmup_steps):
  """Returns `model_fn` closure for TPUEstimator."""
  def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
    """The `model_fn` for TPUEstimator."""

    input_ids = features["input_ids"]
    input_mask = features["input_mask"]
    segment_ids = features["segment_ids"]
    label_ids = features["label_ids"]

    is_predicting = (mode == tf.estimator.ModeKeys.PREDICT)

    # TRAIN and EVAL
    if not is_predicting:

      (loss, predicted_labels, log_probs) = create_model(
        is_predicting, input_ids, input_mask, segment_ids, label_ids, num_labels)

      train_op = bert.optimization.create_optimizer(
          loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu=False)

      # Calculate evaluation metrics.
      def metric_fn(label_ids, predicted_labels):
        accuracy = tf.metrics.accuracy(label_ids, predicted_labels)
        f1_score = tf.contrib.metrics.f1_score(
            label_ids,
            predicted_labels)
        auc = tf.metrics.auc(
            label_ids,
            predicted_labels)
        recall = tf.metrics.recall(
            label_ids,
            predicted_labels)
        precision = tf.metrics.precision(
            label_ids,
            predicted_labels)
        true_pos = tf.metrics.true_positives(
            label_ids,
            predicted_labels)
        true_neg = tf.metrics.true_negatives(
            label_ids,
            predicted_labels)
        false_pos = tf.metrics.false_positives(
            label_ids,
            predicted_labels)
        false_neg = tf.metrics.false_negatives(
            label_ids,
            predicted_labels)
        return {
            "eval_accuracy": accuracy,
            "f1_score": f1_score,
            "auc": auc,
            "precision": precision,
            "recall": recall,
            "true_positives": true_pos,
            "true_negatives": true_neg,
            "false_positives": false_pos,
            "false_negatives": false_neg
        }

      eval_metrics = metric_fn(label_ids, predicted_labels)

      if mode == tf.estimator.ModeKeys.TRAIN:
        return tf.estimator.EstimatorSpec(mode=mode,
          loss=loss,
          train_op=train_op)
      else:
          return tf.estimator.EstimatorSpec(mode=mode,
            loss=loss,
            eval_metric_ops=eval_metrics)
    else:
      (predicted_labels, log_probs) = create_model(
        is_predicting, input_ids, input_mask, segment_ids, label_ids, num_labels)

      predictions = {
          'probabilities': log_probs,
          'labels': predicted_labels
      }
      return tf.estimator.EstimatorSpec(mode, predictions=predictions)

  # Return the actual model function in the closure
  return model_fn


# Compute train and warmup steps from batch size
# These hyperparameters are copied from this colab notebook (https://colab.sandbox.google.com/github/tensorflow/tpu/blob/master/tools/colab/bert_finetuning_with_cloud_tpus.ipynb)
BATCH_SIZE = 64
LEARNING_RATE = 2e-5
NUM_TRAIN_EPOCHS = 4.0
# Warmup is a period of time where hte learning rate
# is small and gradually increases--usually helps training.
WARMUP_PROPORTION = 0.1
# Model configs
SAVE_CHECKPOINTS_STEPS = 1000
SAVE_SUMMARY_STEPS = 1000

# Compute # train and warmup steps from batch size
#num_train_steps = int(len(train_features) / BATCH_SIZE * NUM_TRAIN_EPOCHS)
num_train_steps = 150000
num_warmup_steps = int(num_train_steps * WARMUP_PROPORTION)
def serving_input_fn():
    label_ids   = tf.placeholder(tf.int32, [None], name='label_ids')
    input_ids   = tf.placeholder(tf.int32, [None, MAX_SEQ_LENGTH], name='input_ids')
    input_mask  = tf.placeholder(tf.int32, [None, MAX_SEQ_LENGTH], name='input_mask')
    segment_ids = tf.placeholder(tf.int32, [None, MAX_SEQ_LENGTH], name='segment_ids')
    input_fn = tf.estimator.export.build_raw_serving_input_receiver_fn({
        'label_ids': label_ids,
        'input_ids': input_ids,
        'input_mask': input_mask,
        'segment_ids': segment_ids,
    })()
    return input_fn

def _acc_smaller(best_eval_result, current_eval_result):

    print('best_eval_results:', best_eval_result)
    print('current_eval_results:', current_eval_result)

    return best_eval_result['eval_accuracy'] < current_eval_result['eval_accuracy']


def predict(sentences, predict_fn):
    labels = [0, 1]
    input_examples = [
        run_classifier.InputExample(
            guid="",
            text_a = x,
            text_b = None,
            label = 0
        ) for x in sentences] # here, "" is just a dummy label
    input_features = run_classifier.convert_examples_to_features(
        input_examples, labels, MAX_SEQ_LENGTH, tokenizer
    )

    all_input_ids = []
    all_input_mask = []
    all_segment_ids = []
    all_label_ids = []

    for feature in input_features:
        all_input_ids.append(feature.input_ids)
        all_input_mask.append(feature.input_mask)
        all_segment_ids.append(feature.segment_ids)
        all_label_ids.append(feature.label_id)

    pred_dict = {
        'input_ids': all_input_ids,
        'input_mask': all_input_mask,
        'segment_ids': all_segment_ids,
        'label_ids': all_label_ids
    }

    predictions = predict_fn(pred_dict)
    return [
        (sentence, prediction, label)
        for sentence, prediction, label in zip(pred_sentences, predictions['probabilities'], predictions['labels'])
    ]


print('Predictions from best model')
tf.logging.info('Predictions from best model')

from tensorflow.contrib import predictor

saved_model_dir = 'robertes'

subfolders = [ f.path for f in os.scandir(saved_model_dir) if f.is_dir() ]
total_val_pred_probs = []
total_test_pred_probs = []

y_val=[]
for index, row in valSet.iterrows():
    y_val.append(row['sentiment'])

X = []
X_test = []

for folder in subfolders:
    X_pd = pd.read_csv(folder + "_X_predictions.csv")
    X_test_pd = pd.read_csv(folder + "_X_test_predictions.csv")

    X_tmp = [ast.literal_eval(x) for x in X_pd['X']]
    X_test_tmp = [ast.literal_eval(x) for x in X_test_pd['X']]

    X.append(X_tmp)
    X_test.append(X_test_tmp)


    found_labels = list(np.argmax(X_tmp, axis=1))
    # print('found_labels:',found_labels)
    # print('y_val:', y_val)
    print('model:', folder, 'accuracy_score:', accuracy_score(found_labels, y_val))



x_val = np.hstack(X)
x_test = np.hstack(X_test)


xgb_clf = xgb.XGBClassifier(n_estimators=1, eta=0.05, max_depth=3).fit(x_val, y_val)
y_val_pred = xgb_clf.predict(x_val)
print("xgboost test accuracy:", accuracy_score(y_val, y_val_pred))

print("---------------------- Grid Search ---------------------------------------")
xgb_model = xgb.XGBClassifier(nthread=-1)

parameters = {
        'max_depth': [1, 2, 3, 5, 9],
        'eta' : [0.01, 0.05, 0.14],
        'n_estimators': [20 ,35, 50, 75, 100]
}

clf = GridSearchCV(xgb_model, parameters, n_jobs=-1, cv=10, scoring=make_scorer(accuracy_score), verbose=1)
clf.fit(x_val, y_val)

print("Best estimator:")
print(clf.best_estimator_)
print("Best params:")
print(clf.best_params_)
print("Best score xgboost:")
print(clf.best_score_)

y_val_pred = clf.best_estimator_.predict(x_val)
print("cv xgboost train accuracy with best estimator:", accuracy_score(y_val, y_val_pred))

best_test_pred = clf.best_estimator_.predict(x_test)

idx_list = range(1, len(best_test_pred) + 1)
df = pd.DataFrame(list(zip(idx_list, best_test_pred)), columns =['Id', 'Prediction'])
df = df.replace(0, -1)
print(df)

out_file = folder + "xgb_ensemble.csv"
print('writing predictions for test set in', out_file)
df.to_csv(out_file, index=False)
