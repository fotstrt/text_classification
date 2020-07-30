#!/bin/bash

# Note that this script uses GNU-style sed. On Mac OS, you are required to first
#    brew install gnu-sed --with-default-names
cat ./interm_data/train_pos_full_preprocessed_1_no_dupes.txt ./interm_data/train_neg_full_preprocessed_1_no_dupes.txt | sed "s/ /\n/g" | grep -v "^\s*$" | sort | uniq -c > ./interm_data/vocab_preprocessed_1.txt
