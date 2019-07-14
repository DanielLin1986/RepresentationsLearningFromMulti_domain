# -*- coding: utf-8 -*-
"""
Created on Sun Apr 15 12:20:53 2018

@author: yuyu-

Obtain representations from the network which was trained using CVE/SARD data sets.

"""
from random import randrange
import os
import csv
import pickle
import numpy as np
import pandas as pd

import tensorflow as tf

from keras.layers import Input
from keras.models import load_model, Model

from keras.preprocessing.sequence import pad_sequences
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ------------------------------------------------------------ #
# Parameters used and data directories

LOSS_FUNCTION = 'binary_crossentropy'
#OPTIMIZER = 'adamax'
OPTIMIZER = 'sgd'

project_name = 'ffmpeg'

MAX_LEN = 1000 # The Padding Length for each sample.
EMBEDDING_DIM = 100 # The Embedding Dimension for each element within the sequence of a data sample.

working_dir = 'D:\\Path\\to\\the_project\\' + project_name  + os.sep

model_saved_path = 'D:\\Path\\to\\models\\'

model_name = 'BiLSTM_sard_pretrained_all_tokens_97_1.000_0.000679.h5'

sample_data_dir = 'D:\\Path\\to\\data\\'

#--------------------------------------------------------#
# 2. Load the data which will be the inputs for obtaining the representations.

def getData(file_path):
	
	# The encoding has to be 'latin1' to make sure the string to float convertion to be smooth
	df = pd.read_csv(file_path, sep=',', encoding='latin1', low_memory=False, header=None) 
	df_list = df.values.tolist()
	
	temp = []
	id_list = []
	for i in df_list:
		# Get rid of 'NaN' values.
		i = [x for x in i if str(x) != 'nan']
		i = [str(x) for x in i]
		if len(i) > 3:
			id_element = i[0] # Get the first element whichi is the id
			temp.append(i[1:]) # Get the rest of the elements which are the nodes of ASTs.
			id_list.append(id_element)
	
	return temp, id_list

# Further split the elements such as "const int *" into "const", "int" and "*"
def ProcessList_1(list_to_process):
	token_list = []
	for sub_list_to_process in list_to_process:
		sub_token_list = []
		for each_word in sub_list_to_process:
			sub_word = each_word.split()
			for element in sub_word:
				sub_token_list.append(element)
		token_list.append(sub_token_list)
	return token_list

def LoadSavedData(path):
	with open(path, 'rb') as f:
		loaded_data = pickle.load(f)
	return loaded_data

def processList(list_to_process):
	temp = []
	for i in list_to_process:
		i = [x for x in i if str(x) != ';'] # Remove the ';' in the ffmpeg list.
		temp.append(i)
	
	return temp

def storeOuput(arr, path):
	with open(path, 'w') as myfile:
		wr = csv.writer(myfile)
		wr.writerow(arr)
		
def ListToCSV(list_to_csv, path):
	df = pd.DataFrame(list_to_csv)
	df.to_csv(path, index=False)

def GenerateLabels(input_arr):
	temp_arr = []
	for func_id in input_arr:
		temp_sub_arr = []
		if "cve" in func_id or "CVE" in func_id:
			temp_sub_arr.append(1)
		else:
			temp_sub_arr.append(0)
		temp_arr.append(temp_sub_arr)
	return np.asarray(temp_arr)
#
ffmpeg_test_list = LoadSavedData(working_dir + 'test_file_list.pkl')
ffmpeg_test_list_id = LoadSavedData(working_dir + 'test_file_list_id.pkl')

ffmpeg_train_list = LoadSavedData(working_dir + 'train_file_list.pkl')
ffmpeg_train_list_id = LoadSavedData(working_dir + 'train_file_list_id.pkl')

#ffmpeg_test_list, ffmpeg_test_list_id = getData(working_dir + 'ffmpeg_test.csv')
ffmpeg_test_label = GenerateLabels(ffmpeg_test_list_id)
#
#ffmpeg_train_list, ffmpeg_train_list_id = getData(working_dir + 'ffmpeg_train.csv')
ffmpeg_train_label = GenerateLabels(ffmpeg_train_list_id)

ffmpeg_test_list = ProcessList_1(ffmpeg_test_list)
ffmpeg_test_list = processList(ffmpeg_test_list)
ffmpeg_train_list = ProcessList_1(ffmpeg_train_list)
ffmpeg_train_list = processList(ffmpeg_train_list)

total_list = ffmpeg_train_list + ffmpeg_test_list
total_list_id = ffmpeg_train_list_id + ffmpeg_test_list_id
total_list_label = ffmpeg_train_label.tolist() + ffmpeg_test_label.tolist()
total_list_label = np.ndarray.flatten(np.asarray(total_list_label))

ffmpeg_vul_list = []
ffmpeg_non_vul_list = []
ffmpeg_non_vul_list_id = []
ffmpeg_vul_list_id = []

#print ("The number of training samples are: " + str(len(ffmpeg_train_list)) + ", ID: " + str(len(ffmpeg_train_list_id)) + ", Label: " + str(len(ffmpeg_train_list_id)))
#print ("The number of testing samples are: " + str(len(ffmpeg_test_list)) + ", ID: " + str(len(ffmpeg_test_list_id)) + ", Label: " + str(len(ffmpeg_test_list_id)))
#
#print (np.count_nonzero(ffmpeg_train_label), np.count_nonzero(ffmpeg_test_label))
#--------------------------------------------------------#
# 3. Load the saved model and compile it.

model = load_model(model_saved_path + model_name)

model.compile(loss= LOSS_FUNCTION,
			  optimizer=OPTIMIZER,
			  metrics=['accuracy'])

print ("The model has been loaded: ")
print (model.summary())

"""
Layer (type)                 Output Shape              Param #   
=================================================================
input_2 (InputLayer)         (None, 1000)              0         
_________________________________________________________________
embedding_2 (Embedding)      (None, 1000, 100)         52366800  
_________________________________________________________________
bidirectional_3 (Bidirection (None, 1000, 128)         84992     
_________________________________________________________________
dropout_3 (Dropout)          (None, 1000, 128)         0         
_________________________________________________________________
bidirectional_4 (Bidirection (None, 1000, 128)         99328     
_________________________________________________________________
global_max_pooling1d_2 (Glob (None, 128)               0         
_________________________________________________________________
dropout_4 (Dropout)          (None, 128)               0         
_________________________________________________________________
dense_3 (Dense)              (None, 64)                8256      
_________________________________________________________________
dense_4 (Dense)              (None, 1)                 65        
=================================================================
Total params: 52,559,441
Trainable params: 192,641
Non-trainable params: 52,366,800
_________________________________________________________________

"""

#------------------------------------#
# 4. Load pre-trained word2vec and tokens

print ("Applying tokenization....")

def JoinSubLists(list_to_join):
	new_list = []
	
	for sub_list_token in list_to_join:
		new_line = ','.join(sub_list_token)
		new_list.append(new_line)
	return new_list

ffmpeg_train_list_new = JoinSubLists(ffmpeg_train_list)
ffmpeg_test_list_new = JoinSubLists(ffmpeg_test_list)

#non_vul_list_new = JoinSubLists(cwe_non_vul_rdm)
#vul_list_new = JoinSubLists(cwe_vul_rdm)

tokenizer = LoadSavedData('D:\\Path\\to\\tokenizer.pickle')
ffmpeg_train_sequences = tokenizer.texts_to_sequences(ffmpeg_train_list_new)
ffmpeg_test_sequences = tokenizer.texts_to_sequences(ffmpeg_test_list_new)

#non_vul_sequences = tokenizer.texts_to_sequences(non_vul_list_new)
#vul_sequences = tokenizer.texts_to_sequences(vul_list_new)

word_index = tokenizer.word_index
print ('Found %s unique tokens.' % len(word_index))

print ("The length of tokenized training sequences: " + str(len(ffmpeg_train_sequences)))
print ("The length of tokenized testing sequences: " + str(len(ffmpeg_test_sequences)))

print ("Loading trained word2vec model...")

# Load the pre-trained embeddings.
w2v_model_path = 'D:\\Path\\to\\w2v_model.txt'
w2v_model = open(w2v_model_path, encoding="latin1")

print ("----------------------------------------")
print ("The trained word2vec model: ")
print (w2v_model)

#------------------------------------#
# 3. Do the paddings.
print ("max_len ", MAX_LEN)
print('Pad sequences (samples x time)')

ffmpeg_train_sequences_pad = pad_sequences(ffmpeg_train_sequences, maxlen = MAX_LEN, padding ='post')
ffmpeg_test_sequences_pad = pad_sequences(ffmpeg_test_sequences, maxlen = MAX_LEN, padding ='post')
print (ffmpeg_train_sequences_pad.shape)
print (ffmpeg_test_sequences_pad.shape)

#non_vul_sequences_pad = pad_sequences(non_vul_sequences, maxlen = MAX_LEN, padding ='post')
#vul_sequences_pad = pad_sequences(vul_sequences, maxlen = MAX_LEN, padding ='post')
#print (non_vul_sequences_pad.shape)
#print (vul_sequences_pad.shape)

# Acquire the embeddings.
def ObtainRepresentations(input_sequences, layer_number, model):
	layered_model = Model(inputs = model.input, outputs=model.layers[layer_number].output)
	representations = layered_model.predict(input_sequences)
	return representations

# There are 9 layers in total. So we use the fourth last layer
ffmpeg_train_repre_nist = ObtainRepresentations(ffmpeg_train_sequences_pad, 5, model) 
ffmpeg_test_repre_nist = ObtainRepresentations(ffmpeg_test_sequences_pad, 5, model) 

#non_vul_repre_nist = ObtainRepresentations(non_vul_sequences_pad, 8, model) 
#vul_repre_nist = ObtainRepresentations(vul_sequences_pad, 8, model) 
#
#total_list = non_vul_repre_nist.tolist() + vul_repre_nist.tolist() + ffmpeg_non_vul_repre_nist.tolist() + ffmpeg_vul_repre_nist.tolist()
#total_list_label = cwe_non_vul_rdm_label + cwe_vul_rdm_label + ffmpeg_non_vul_rdm_label + ffmpeg_vul_rdm_label
#total_list_id = cwe_non_vul_rdm_id + cwe_vul_rdm_id + ffmpeg_non_vul_rdm_id + ffmpeg_vul_rdm_id
#
#result_set = TSNE(n_components=2, perplexity=20).fit_transform(total_list)
#result_set_X = result_set[:, 0]
#result_set_y = result_set[:, 1]
##color_list = ['red' if label == 1 else 'green' for label in total_list_label]
#color_list = []
#
#for label_item in total_list_label:
#    if label_item == 1:
#        color_list.append('red')
#    if label_item == 0:
#        color_list.append('green')
#    if label_item == 2:
#        color_list.append('blue')
#    if label_item == 3:
#        color_list.append('purple')
#    
#
#fig = plt.figure(figsize=(9,9))
#plt.scatter(result_set_X, result_set_y, color = color_list)
#green_patch = mpatches.Patch(color='green', label='Non-vul. in CWE Synthetic Data')
#red_patch = mpatches.Patch(color='red', label='Vul. in CWE Synthetic Data')
#blue_patch = mpatches.Patch(color='blue', label='Non-vul. in FFmpeg Data')
#purple_patch = mpatches.Patch(color='purple', label='Vul. in FFmpeg Data')
#
#plt.legend(handles = [red_patch, green_patch, blue_patch, purple_patch], loc=1)
#plt.title('The plot of the representations of the randomly selected data from vul. and non-vul. dataset.')

np.savetxt(working_dir + 'ffmpeg_train_nist_rep_128.csv', ffmpeg_train_repre_nist, delimiter=",")
np.savetxt(working_dir + 'ffmpeg_train_nist_label.csv', ffmpeg_train_label, delimiter=",")
np.savetxt(working_dir + 'ffmpeg_test_nist_rep_128.csv', ffmpeg_test_repre_nist, delimiter=",")
np.savetxt(working_dir + 'ffmpeg_test_nist_label.csv', ffmpeg_test_label, delimiter=",")

ListToCSV(ffmpeg_test_list_id, working_dir + 'ffmpeg_test_id.csv')
ListToCSV(ffmpeg_train_list_id, working_dir + 'ffmpeg_train_id.csv')

#------------------------------------------------------------------------------
# Get the ids of the data that satisfy certain requirements
#set_1 = []
#
#for index, item in enumerate(result_set):
#    if item[0] < 9 and item[0] > 0 and item[1] > 40 and item [1] < 46:
#        set_1.append(total_list_id[index])
