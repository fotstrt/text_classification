#!/bin/bash

# Note that this script uses GNU-style sed. On Mac OS, you are required to first
#    brew install gnu-sed --with-default-names
cat ./interm_data/vocab_preprocessed_1_no_dupes.txt | sed "s/^\s\+//g" | sort -rn | grep -v "^[1234]\s" | cut -d' ' -f2 > ./interm_data/vocab_cut_preprocessed_1_no_dupes.txt
