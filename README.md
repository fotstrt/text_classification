# Applying Ensembling Methods for Sentiment Classification of Tweets

## Datasets creation

### To download data:
- `wget http://www.da.inf.ethz.ch/teaching/2018/CIL/material/exercise/twitter-datasets.zip`

### To prepare datasets:

- `python3 infrastructure/convert.py` prepares the train and test datasets merging positive and negative tweets with the appropriate labels (1: positive tweets, 0: negative tweets).

### To preprocess data:

- `python3 preprocessing/preprocess_punctuation.py 'path-to-csv-data' 0/1` (0: training data preprocessing, 1: test data preprocessing


## Baseline Implementations

The baseline models can be found in the `implementations/baselines` folder.

To download the essential data for this project:
- `mkdir data`
- `cd data`
- `wget http://www.da.inf.ethz.ch/teaching/2018/CIL/material/exercise/twitter-datasets.zip` (to download dataset)
- `wget` Twitter pretrained embeddings from https://nlp.stanford.edu/projects/glove/
- `mkdir interm_data`
- `mkdir final_data`

To build a co-occurence matrix, run the following commands:

Note that the cooc.py script takes a few minutes to run, and displays the number of tweets processed.

- `./infrastructure/build_vocab.sh`
- `./infrastructure/cut_vocab.sh`
- `python3 ./infrastructure/pickle_vocab.py`
- `python3 ./infrastructure/cooc.py`

(Not mandatory) For preprocessing:

- `python3 ./preprocessing/preprocess_baselines.py 'path-to-txt-data' 0/1 (for each data file seperately, 0: training data preprocessing, 1: test data preprocessing)`

(Not mandatory) For removing duplicates:

- `python3 ./infrastructure/deduplication.py`

For manually computing GloVe embeddings: (change number of dimensions)

- `python3 ./infrastructure/glove_compute.py`

(Not mandatory) For pretrained GloVe embeddings: (needs previous manual computation of embeddings)

- `python3 ./infrastructure/glove_pretrained.py`

For computing tweet embeddings:

- `python3 ./infrastructure/infrastructure.py` (for manual embeddings)
- `python3 ./infrastructure/infrastructure_pretrained.py` (for pretrained embeddings)

For classification task:

- `python3 ./implementations/baselines/svm.py`
- `python3 ./implementations/baselines/xgboost_impl.py`
- `python3 ./implementations/baselines/logistic.py`

## LSTM-based models

## BERT-based models

The implementations regarding to **BERT** are contained in the `implementations/bert` folder. There can be found the code scripts for:

1. A pretrained **BERT** model in `bert.py`
1. A pretrained **BERT** model using hugging faces in `bert.py`

## Ensembling methods

We used two ensembling methods in order to optimize our final model:

1. Linear regression
1. XGBoost

The respective implementations can be found in `implementations/ensembling` folder.
