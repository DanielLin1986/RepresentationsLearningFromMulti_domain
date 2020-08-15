# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 21:54:21 2017

@author: yuyu-

Train a word2vec model using all the functions from 6 projects.

"""

import time
import pickle
import os
import pandas as pd

from gensim.models import Word2Vec
from keras.preprocessing.text import Tokenizer
from LoadCFilesAsText import getCFilesFromText

script_start_time = time.time()

print ("Script starts at: " + str(script_start_time))

working_dir = "D:\\Your\\Directory\\"

file_path = working_dir + "Path\\to\\Your\\Source\\Code\\"

#--------------------------------------------------------#
# 1. Load data from .csv files. The data contains all the functions from 6 projects.

# Get the csv file and convert it to a list.
def getData(filePath):
    df = pd.read_csv(filePath, sep=",", low_memory=False)
    df_list = df.values.tolist()
    temp = []
    ####
    for i in df_list:
        # Get rid of 'NaN' values.
        i = [x for x in i if str(x) != 'nan']
        temp.append(i)
    
    return temp

# Further split the elements such as "const int *" into "const", "int" and "*"
def ProcessList(list_to_process):
    token_list = []
    for sub_list_to_process in list_to_process:
        sub_token_list = []
        for each_word in sub_list_to_process:
            sub_word = each_word.split()
            for element in sub_word:
                sub_token_list.append(element)
        token_list.append(sub_token_list)
    return token_list


#train_token_list = getData(file_path)
train_token_list, train_token_list_id = getCFilesFromText(file_path)

train_token_list = ProcessList(train_token_list)

print ("The length of training set is : " + str(len(train_token_list)))

#--------------------------------------------------------#
# 2. Tokenization: convert the loaded text (the nodes of ASTs) to tokens.

new_total_token_list = []

for sub_list_token in train_token_list:
    new_line = ','.join(sub_list_token)
    new_total_token_list.append(new_line)

tokenizer = Tokenizer(num_words=None, filters=',', lower=False, char_level=False)
tokenizer.fit_on_texts(new_total_token_list)

# Save the tokenizer.
with open('tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

# ----------------------------------------------------- #
# 3. Train a Vocabulary with Word2Vec -- using the function provided by gensim

#w2vModel = Word2Vec(train_token_list, workers = 12, size=100) # With default settings, the embedding dimension is 100 and using, (sg=0), CBOW is used.  
w2vModel = Word2Vec(train_token_list, workers = 12, size=100, sg=1)

print ("----------------------------------------")
print ("The trained word2vec model: ")
print (w2vModel)

w2vModel.wv.save_word2vec_format(working_dir + "6_projects_w2v_model_skipgram.txt", binary=False)
