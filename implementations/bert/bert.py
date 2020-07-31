import numpy as np
import pandas as pd
import os
import re
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
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import gensim
from gensim.models.keyedvectors import KeyedVectors
from keras.models import Sequential, Model
from keras.regularizers import l1, l2
from sklearn.metrics import classification_report
import tensorflow as tf
import tensorflow_estimator
import tensorflow_hub as hub
from datetime import datetime

#!pip install bert-tensorflow

import tensorflow as tf
print(tf.__version__)

import bert
from bert import run_classifier
from bert import optimization
from bert import tokenization
import warnings

import logging

# get TF logger
log = logging.getLogger('tensorflow')
log.setLevel(logging.INFO)

# create formatter and add it to the handlers
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# create file handler which logs even debug messages
fh = logging.FileHandler('tensorflow.log')
fh.setLevel(logging.DEBUG)
fh.setFormatter(formatter)
log.addHandler(fh)

tf.logging.set_verbosity(tf.logging.INFO)

data = pd.read_csv('csv_data/train_data_full-not-processed.csv',
                   index_col = 0,
                   names=["id", "sentiment", "tweet"])
data = data.dropna()
print(data.head(5))


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

# Data Preprocessing for BERT
# Use the InputExample class from BERT's run_classifier code to create examples from the data
train_InputExamples = trainingSet.apply(lambda x: bert.run_classifier.InputExample(guid=None, # Globally unique ID for bookkeeping, unused in this example
                                                                       text_a = x[DATA_COLUMN],
                                                                       text_b = None,
                                                                       label = x[LABEL_COLUMN]), axis = 1)

val_InputExamples = valSet.apply(lambda x: bert.run_classifier.InputExample(guid=None,
                                                                       text_a = x[DATA_COLUMN],
                                                                       text_b = None,
                                                                       label = x[LABEL_COLUMN]), axis = 1)

# test_InputExamples = testSet.apply(lambda x: bert.run_classifier.InputExample(guid=None,
#                                                                       text_a = x[DATA_COLUMN],
#                                                                       text_b = None,
#                                                                       label = x[LABEL_COLUMN]), axis = 1)

# This is a path to an uncased (all lowercase) version of BERT
# BERT_MODEL_HUB = "https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1"
BERT_MODEL_HUB = '/cluster/scratch/atriantaf/twitter-sentiment-analysis/bert-model-hub-large/'
# BERT_MODEL_HUB = "https://tfhub.dev/google/bert_uncased_L-24_H-1024_A-16/1"

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
MAX_SEQ_LENGTH = 64
# Convert our train, validation and test features to InputFeatures that BERT understands.
train_features = bert.run_classifier.convert_examples_to_features(train_InputExamples, label_list, MAX_SEQ_LENGTH, tokenizer)
val_features = bert.run_classifier.convert_examples_to_features(val_InputExamples, label_list, MAX_SEQ_LENGTH, tokenizer)
# test_features = bert.run_classifier.convert_examples_to_features(test_InputExamples, label_list, MAX_SEQ_LENGTH, tokenizer)

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
NUM_TRAIN_EPOCHS = 3.0
# Warmup is a period of time where hte learning rate
# is small and gradually increases--usually helps training.
WARMUP_PROPORTION = 0.1
# Model configs
SAVE_CHECKPOINTS_STEPS = 1000
SAVE_SUMMARY_STEPS = 1000

# Compute # train and warmup steps from batch size
num_train_steps = int(len(train_features) / BATCH_SIZE * NUM_TRAIN_EPOCHS)
#num_train_steps = 120000
num_warmup_steps = int(num_train_steps * WARMUP_PROPORTION)

int(len(train_features) / BATCH_SIZE * NUM_TRAIN_EPOCHS)

model_fn = model_fn_builder(
  num_labels=len(label_list),
  learning_rate=LEARNING_RATE,
  num_train_steps=num_train_steps,
  num_warmup_steps=num_warmup_steps)

estimator = tf.estimator.Estimator(
  model_fn=model_fn,
  config=tf.estimator.RunConfig(model_dir='./bert_model_not_processed_0_1_val_set_seq_len_64', save_summary_steps=SAVE_SUMMARY_STEPS,
                                save_checkpoints_steps=SAVE_CHECKPOINTS_STEPS), params={"batch_size": BATCH_SIZE})

# Create an input function for training. drop_remainder = True for using TPUs.
train_input_fn = bert.run_classifier.input_fn_builder(
    features=train_features,
    seq_length=MAX_SEQ_LENGTH,
    is_training=True,
    drop_remainder=False)

val_input_fn = run_classifier.input_fn_builder(features=val_features, seq_length=MAX_SEQ_LENGTH,
                                                is_training=False, drop_remainder=False)

# test_input_fn = run_classifier.input_fn_builder(features=test_features, seq_length=MAX_SEQ_LENGTH,
#                                                 is_training=False, drop_remainder=False)

train_spec = tf.estimator.TrainSpec(input_fn=train_input_fn, max_steps=num_train_steps)
# train_spec = tf.estimator.TrainSpec(input_fn=train_input_fn, max_steps=500)

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

    return best_eval_result['eval_accuracy'] < current_eval_result['eval_accuracy']

exporter = tf.estimator.BestExporter(exports_to_keep=5, serving_input_receiver_fn=serving_input_fn,
                                     compare_fn=_acc_smaller)

eval_spec = tf.estimator.EvalSpec(input_fn=val_input_fn, steps=3000, exporters=exporter, start_delay_secs=0,
                                  throttle_secs=5)

# warnings.filterwarnings("ignore")
print('Beginning Training!')
current_time = datetime.now()
print("num_train_steps:", num_train_steps)
tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)
print("Training took time ", datetime.now() - current_time)
"""
print('Evaluating on test set!')
tf.logging.info('Evaluating on test set!')
estimator.evaluate(input_fn=test_input_fn, steps=None, checkpoint_path=None)

print('Predictions from best model')
tf.logging.info('Predictions from best model')

from tensorflow.contrib import predictor

saved_model_dir = 'my_bert_model/export/best_exporter'
subfolders = [ f.path for f in os.scandir(saved_model_dir) if f.is_dir() ]
saved_model_folder = subfolders[0]

predict_fn = predictor.from_saved_model(saved_model_folder)

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


pred_sentences = list(valSet["tweet"])
predictions = []
for idx in range(0, len(pred_sentences), 100):
    predictions += predict(pred_sentences[idx:idx+100], predict_fn)

pred = [p for _,_,p in predictions]
true = list(valSet["sentiment"])

print("val accuracy:", accuracy_score(true, pred))
tf.logging.info("val accuracy: %f", accuracy_score(true, pred))

pred_sentences = list(testSet["tweet"])
predictions = []
for idx in range(0, len(pred_sentences), 100):
    predictions += predict(pred_sentences[idx:idx+100], predict_fn)

pred = [p for _,_,p in predictions]
true = list(testSet["sentiment"])

print("test accuracy:", accuracy_score(true, pred))
tf.logging.info("test accuracy: %f", accuracy_score(true, pred))

# # check for test accuracy by giving just features (not labels) and then compute labels

# data_new = pd.read_csv(TEST_PROCESSED_FILE, header=None,  index_col=0)
# print(data_new.shape)
# data_new.columns=["Sentence"]
# idx = data_new[data_new['Sentence'].isnull() == True]
# #print(idx)

# data_new.at[6124,'Sentence']='love' # TODO: change this!
# #print(data_new)
"""
