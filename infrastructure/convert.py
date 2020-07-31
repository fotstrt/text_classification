import pandas as pd
import numpy as np

input_path = "../csv_data/"
output_path = "../csv_data/"


# the format of the train data should be: tweet_id,sentiment,tweet
# the format of the test data should be: tweet_id,tweet
# sentiment is 1(positive) or 0(negative)
# tweet should be in ""

it=1

with open(output_path+"train_data.csv", "w", encoding="utf-8") as fw:

    with open(input_path+"train_pos_embeddings.csv", "r", encoding="utf-8") as fr1:
        line = fr1.readline()
        while line:
            new_line = str(it) + ",1," + '"' + line[:-1] + '"' + "\n"
            fw.writelines(new_line)
            it+=1

    with open(input_path+"train_neg_embeddings.csv", "r", encoding="utf-8") as fr2:
        line = fr2.readline()
        while line:
            new_line = str(it) + ",0," + '"' + line[:-1] + '"' + "\n"
            fw.writelines(new_line)
            it+=1

with open(output_path+"test_data.csv", "w", encoding="utf-8") as fw:

    with open(input_path+"test_data_embeddings.csv", "r", encoding="utf-8") as fr:
        line = fr.readline()
        while line:
            tokens=line.split(',')
            new_line = tokens[0] + "," + '"'
            for i in range(1, len(tokens)):
                if (i==len(tokens)-1):
                    new_line+=tokens[i][:-1]
                else:
                    new_line+=tokens[i]
            new_line +=  '"\n'
            fw.writelines(new_line)
            line = fr.readline()