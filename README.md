# text_classification
Text Sentiment Classification based on tweets

## Build the Co-occurence Matrix

To build a co-occurence matrix, run the following commands.  (Remember to put the data files
in the correct locations)

Note that the cooc.py script takes a few minutes to run, and displays the number of tweets processed.

- `build_vocab.sh`
- `cut_vocab.sh`
- `python3 pickle_vocab.py`
- `python3 cooc.py`

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

