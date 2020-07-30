import pandas as pd
file_name_1 = "./interm_data/train_pos_full_preprocessed_1.txt"
file_name_2 = "./interm_data/train_neg_full_preprocessed_1.txt"
file_name_output_1 = "./interm_data/train_pos_full_preprocessed_1_no_dupes.txt"
file_name_output_2 = "./interm_data/train_neg_full_preprocessed_1_no_dupes.txt"


lines = open(file_name_1, 'r').readlines()
lines_set = set(lines)

out  = open(file_name_output_1, 'w')

for line in lines_set:
    out.write(line)

lines = open(file_name_2, 'r').readlines()
lines_set = set(lines)

out = open(file_name_output_2, 'w')

for line in lines_set:
    out.write(line)

# Notes:
# - the `subset=None` means that every column is used
#    to determine if two rows are different; to change that specify
#    the columns as an array
# - the `inplace=True` means that the data structure is changed and
#   the duplicate rows are gone
##df_1.drop_duplicates(subset=None, inplace=True)
##df_2.drop_duplicates(subset=None, inplace=True)

# Write the results to a different file
##df_1.to_csv(file_name_output_1, index=False)
##df_2.to_csv(file_name_output_2, index=False)
