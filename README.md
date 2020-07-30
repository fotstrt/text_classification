# text_classification
Text Sentiment Classification based on tweets

## Baseline Implementations

In the root directory: (Remember to change the name of files and put the data files
in the correct locations!!!)


- `mkdir data`
- `In /data wget dataset from http://www.da.inf.ethz.ch/teaching/2018/CIL/material/exercise/twitter-datasets.zip`
- `In /data wget Twitter pretrained embeddings from https://nlp.stanford.edu/projects/glove/`
- 'mkdir interm_data'
- 'mkdir final_data'

To build a co-occurence matrix, run the following commands:

Note that the cooc.py script takes a few minutes to run, and displays the number of tweets processed.

- `./infrastructure/build_vocab.sh`
- `./infrastructure/cut_vocab.sh`
- `python3 ./infrastructure/pickle_vocab.py`
- `python3 ./infrastructure/cooc.py`

(Not mandatory) For preprocessing:

- `python3 ./infrastructure/preprocess_new.py 'path-to-txt-data' 0/1 (for each data file seperately, 0: training data preprocessing, 1: test data preprocessing)`

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

- `python3 ./implementations/svm.py`
- `python3 ./implementations/xgboost_impl.py`
- `python3 ./implementations/logistic.py`

### To connect to the Leonhard cluster:
`ssh username@login.leonhard.ethz.ch`

### To run on GPU
`module load python_gpu/3.6.1 hdf5/1.10.1`

`bsub -I -R "rusage[mem=4096, ngpus_excl_p=1]" "python code/lstm.py"`

### To download data:
http://www.da.inf.ethz.ch/teaching/2018/CIL/material/exercise/twitter-datasets.zip

### To preprocess:
from src/ folder:

    python convert.py  --> creates 2 files to csv_data

    python preprocess.py 'path-to-csv-data' 0/1 (0: training data preprocessing, 1: test data preprocessing

    python stats.py 'path to train-processed-data'

