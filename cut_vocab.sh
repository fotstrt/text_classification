#!/bin/bash

# Note that this script uses GNU-style sed. On Mac OS, you are required to first
#    brew install gnu-sed --with-default-names

# Gets all tokens found in "vocab.txt" and sorts them according to the number of appearances (it does not take into account tokens appeared less than 5 times).
# It creates the file "vocab_cut.txt"

cat vocab.txt | sed "s/^\s\+//g" | sort -rn | grep -v "^[1234]\s" | cut -d' ' -f2 > vocab_cut.txt
