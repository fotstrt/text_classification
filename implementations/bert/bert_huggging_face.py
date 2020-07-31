import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.model_selection import KFold
import datetime

from transformers import BertTokenizer

import logging
import sys

# logging.basicConfig(level=logging.INFO, filename='bert_hug.log')
filename='bert_hug.log'
sys.stdout = open(filename, 'w')

print('before load')
tokenizer = BertTokenizer.from_pretrained('/cluster/scratch/atriantaf/wwm_uncased_L-24_H-1024_A-16', do_lower_case=True)
print('after load')
max_length_test = 64

test_sentence = 'Test tokenization sentence. Followed by another sentence'
# add special tokens
test_sentence_with_special_tokens = '[CLS]' + test_sentence + '[SEP]'
tokenized = tokenizer.tokenize(test_sentence_with_special_tokens)
print('tokenized', tokenized)
# convert tokens to ids in WordPiece
input_ids = tokenizer.convert_tokens_to_ids(tokenized)

# precalculation of pad length, so that we can reuse it later on
padding_length = max_length_test - len(input_ids)
# map tokens to WordPiece dictionary and add pad token for those text shorter than our max length
input_ids = input_ids + ([0] * padding_length)
# attention should focus just on sequence with non padded tokens
attention_mask = [1] * len(input_ids)
# do not focus attention on padded tokens
attention_mask = attention_mask + ([0] * padding_length)
# token types, needed for example for question answering, for our purpose we will just set 0 as we have just one sequence
token_type_ids = [0] * max_length_test
bert_input = {
    "token_ids": input_ids,
    "token_type_ids": token_type_ids,
    "attention_mask": attention_mask
}
print(bert_input)

bert_input = tokenizer.encode_plus(
                        test_sentence,
                        add_special_tokens = True, # add [CLS], [SEP]
                        max_length = max_length_test, # max length of the text that can go to BERT
                        pad_to_max_length = True, # add [PAD] tokens
                        return_attention_mask = True, # add attention mask to not focus on pad tokens
              )
print('encoded', bert_input)

# map to the expected input to TFBertForSequenceClassification, see here
def map_example_to_dict(input_ids, attention_masks, token_type_ids, label):
  return {
      "input_ids": input_ids,
      "token_type_ids": token_type_ids,
      "attention_mask": attention_masks,
  }, label

def encode_examples(ds, limit=-1):
  # prepare list, so that we can build up final TensorFlow dataset from slices.
  input_ids_list = []
  token_type_ids_list = []
  attention_mask_list = []
  label_list = []
  if (limit > 0):
      ds = ds.take(limit)


  for label, review in ds.to_numpy():
    #print(review)
    bert_input = convert_example_to_feature(review)

    input_ids_list.append(bert_input['input_ids'])
    token_type_ids_list.append(bert_input['token_type_ids'])
    attention_mask_list.append(bert_input['attention_mask'])
    label_list.append([label])
  return tf.data.Dataset.from_tensor_slices((input_ids_list, attention_mask_list, token_type_ids_list, label_list)).map(map_example_to_dict)

max_length = 64
batch_size = 64
print(batch_size)

def convert_example_to_feature(review):

  return tokenizer.encode_plus(review,
                add_special_tokens = True,
                max_length = max_length,
                pad_to_max_length = True,
                return_attention_mask = True,
              )


input_path = "../../csv_data/"
TRAIN_PROCESSED_FILE = input_path + 'train_data-processed_full.csv'
TEST_PROCESSED_FILE = input_path + 'test_data-processed.csv'


data = pd.read_csv(TRAIN_PROCESSED_FILE, header=None,  index_col=0)
data.columns=["Label", "Sentence"]
print(data.shape)
print(data[data['Sentence'].isnull() == True])
data = data.dropna()
print(data.shape)


trainingSet, remSet = train_test_split(data, test_size=0.2, random_state=7)
valSet, testSet = train_test_split(remSet, test_size=0.5, random_state=7)

print("--- Training -data --")
ds_train_encoded = encode_examples(trainingSet).shuffle(10000).batch(batch_size)

print("--- Validating data ---")
ds_val_encoded = encode_examples(valSet).batch(batch_size)

print("--- Testing data ---")
ds_test_encoded = encode_examples(testSet).batch(batch_size)


from transformers import TFBertForSequenceClassification

learning_rate = 2e-5
number_of_epochs = 3

tf.get_logger().setLevel('INFO')

# model initialization
model = TFBertForSequenceClassification.from_pretrained('bert-large-uncased')

# classifier Adam recommended
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, epsilon=1e-08)

# we do not have one-hot vectors, we can use sparce categorical cross entropy and accuracy
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')

filepath = "models/bert-large-best.h5"
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=0.000001)

class MetricsCallback(tf.keras.callbacks.Callback):

  def __init__(self):
      self.best_acc = 0

  def on_train_batch_begin(self, batch, logs=None):
    # print('Training: batch {} begins at {}'.format(batch, datetime.datetime.now().time()))
      return

  def on_train_batch_end(self, batch, logs=None):
      if batch % 1500 == 0:
          loss, acc = self.model.evaluate(ds_val_encoded, verbose=2)
          print('eval_accuracy:', acc)
          self.best_acc = max(self.best_acc, acc)
          if acc == self.best_acc:
               print('Saving best model with accuracy:', acc)
               self.model.save_weights(filepath)


  def on_test_batch_begin(self, batch, logs=None):
    # print('Evaluating: batch {} begins at {}'.format(batch, datetime.datetime.now().time()))
      return

  def on_test_batch_end(self, batch, logs=None):
    # print('Evaluating: batch {} ends at {}'.format(batch, datetime.datetime.now().time()))
      return

metrics_callback = MetricsCallback()
model.compile(optimizer=optimizer, loss=loss, metrics=[metric])

print(model.summary())

bert_history = model.fit(ds_train_encoded, batch_size=batch_size, epochs=4, validation_data=ds_val_encoded, shuffle=True, callbacks=[reduce_lr, metrics_callback])

# model.save('tf_model.h5', save_format="tf")
