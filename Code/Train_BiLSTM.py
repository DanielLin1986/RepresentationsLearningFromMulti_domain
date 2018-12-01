# -*- coding: utf-8 -*-
"""
Created on Thu Apr 12 15:18:51 2018

@author: yuyu-

Train a Bi-LSTM model.

"""

import time
import datetime
import os
import numpy as np
import pickle
import pandas as pd
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

from keras import optimizers
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.callbacks import TensorBoard, CSVLogger
from keras.models import load_model, Model
from keras.utils import multi_gpu_model
import keras.backend as K

import matplotlib.pyplot as plt

#from attention import AttentionWithContext

from BiLSTM_model import BiLSTM_network
#from BiLSTM_model import BiLSTM_network_with_attention
from manual_token_list import construc_list
#from utils import visualize_attention
from LoadCFilesAsText import getCFilesFromText
# -------------------------------------------------------
# Parameters used
MAX_LEN = 1000 # The Padding Length for each sample.
EMBEDDING_DIM = 100 # The Embedding Dimension for each element within the sequence of a data sample. 

BATCH_SIZE = 16
EPOCHS = 170
PATIENCE = 65

working_dir = 'D:\\Path\\to\\your\\data\\'
model_saved_path = working_dir + os.sep + 'models'
log_path = working_dir + 'logs'

LOSS_FUNCTION = 'binary_crossentropy'
#OPTIMIZER = 'adamax'
learning_rate = 0.01
decay_rate = learning_rate / EPOCHS
#momentum = 0.8

sgd = optimizers.SGD(lr=learning_rate, decay=decay_rate, nesterov=True)
OPTIMIZER = sgd

# 1. Load the data for training and validation
#--------------------------------------------------------

def LoadSavedData(path):
    with open(path, 'rb') as f:
        loaded_data = pickle.load(f)
    return loaded_data

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

# Remove ';' from the list.
def removeSemicolon(input_list):
    new_list = []
    for line in input_list:
        new_line = []
        for item in line:
            if item != ';' and item != ',':
                new_line.append(item)
        new_list.append(new_line)
    
    return new_list

# Further split the elements further such as "const int *" into "const", "int" and "*"
def ProcessList(list_to_process):
    token_list = []
    empty_index_list = []
    for index, sub_list_to_process in enumerate(list_to_process):
        sub_token_list = []
        if len(sub_list_to_process) != 0:
            for each_word in sub_list_to_process: # Remove the empty row
                each_word = str(each_word)
                sub_word = each_word.split()
                for element in sub_word:
                    sub_token_list.append(element)
            token_list.append(sub_token_list)
        else:
            empty_index_list.append(index)
    return token_list, empty_index_list

def ListToCSV(list_to_csv, path):
    df = pd.DataFrame(list_to_csv)
    df.to_csv(path, index=False)

# When test on project FFmpeg, we use the other 5 projects as training set.    
#------------------------------------------------------------#    
libtiff_list = LoadSavedData(working_dir + '\\TII\\FFmpeg\\5-project-except-ffmpeg\\5-project-funcs\\libtiff_list.pkl')
libtiff_list_id = LoadSavedData(working_dir + '\\TII\\FFmpeg\\5-project-except-ffmpeg\\5-project-funcs\\libtiff_list_id.pkl')
libtiff_label = LoadSavedData(working_dir + '\\TII\\FFmpeg\\5-project-except-ffmpeg\\5-project-funcs\\libtiff_label.pkl')

libpng_list = LoadSavedData(working_dir + '\\TII\\FFmpeg\\5-project-except-ffmpeg\\5-project-funcs\\libpng_list.pkl')
libpng_list_id = LoadSavedData(working_dir + '\\TII\\FFmpeg\\5-project-except-ffmpeg\\5-project-funcs\\libpng_list_id.pkl')
libpng_label = LoadSavedData(working_dir + '\\TII\\FFmpeg\\5-project-except-ffmpeg\\5-project-funcs\\libpng_label.pkl')

pidgin_list = LoadSavedData(working_dir + '\\TII\\FFmpeg\\5-project-except-ffmpeg\\5-project-funcs\\pidgin_list.pkl')
pidgin_list_id = LoadSavedData(working_dir + '\\TII\\FFmpeg\\5-project-except-ffmpeg\\5-project-funcs\\pidgin_list_id.pkl')
pidgin_label = LoadSavedData(working_dir + '\\TII\\FFmpeg\\5-project-except-ffmpeg\\5-project-funcs\\pidgin_label.pkl')

vlc_list = LoadSavedData(working_dir + '\\TII\\FFmpeg\\5-project-except-ffmpeg\\5-project-funcs\\vlc_list.pkl')
vlc_list_id = LoadSavedData(working_dir + '\\TII\\FFmpeg\\5-project-except-ffmpeg\\5-project-funcs\\vlc_list_id.pkl')
vlc_label = LoadSavedData(working_dir + '\\TII\\FFmpeg\\5-project-except-ffmpeg\\5-project-funcs\\vlc_label.pkl')

asterisk_list = LoadSavedData(working_dir + '\\TII\\FFmpeg\\5-project-except-ffmpeg\\5-project-funcs\\asterisk_list.pkl')
asterisk_list_id = LoadSavedData(working_dir + '\\TII\\FFmpeg\\5-project-except-ffmpeg\\5-project-funcs\\asterisk_list_id.pkl')
asterisk_label = LoadSavedData(working_dir + '\\TII\\FFmpeg\\5-project-except-ffmpeg\\5-project-funcs\\asterisk_label.pkl')

total_list = libtiff_list + libpng_list + pidgin_list + vlc_list + asterisk_list
total_list_id = libtiff_list_id + libpng_list_id + pidgin_list_id + vlc_list_id + asterisk_list_id
total_list_label = (np.asarray(libtiff_label).flatten()).tolist() + (np.asarray(libpng_label).flatten()).tolist() + (np.asarray(pidgin_label).flatten()).tolist() + (np.asarray(vlc_label).flatten()).tolist() + (np.asarray(asterisk_label).flatten()).tolist()

total_list, empty_index_list = ProcessList(total_list)
total_list = removeSemicolon(total_list)

# Remove empty list
for item in empty_index_list:
    if 'cve'in total_list_id[item] or 'CVE' in total_list_id[item]:
        print (total_list_id[item])
    del total_list_id[item]
    del total_list_label[item]

# We use FFmpeg as the test set.
#------------------------------------------------------------#

ffmpeg_train_list, ffmpeg_train_list_id = getCFilesFromText(working_dir + '\\TII\\FFmpeg\\5-project-except-ffmpeg\\train_func\\')
ffmpeg_train_label = GenerateLabels(ffmpeg_train_list_id).flatten()

ffmpeg_train_list, ffmpeg_train_empty_list = ProcessList(removeSemicolon(ffmpeg_train_list)) 

ffmpeg_test_list, ffmpeg_test_list_id = getCFilesFromText(working_dir + '\\TII\\FFmpeg\\5-project-except-ffmpeg\\test_func\\')
ffmpeg_test_label = GenerateLabels(ffmpeg_test_list_id).flatten()

ffmpeg_test_list, ffmpeg_test_empty_list = ProcessList(removeSemicolon(ffmpeg_test_list)) 
# 2. Load the Tokenizer and the trained word2vec model
#--------------------------------------------------------

# 2.1 Load the toknizer
def JoinSubLists(list_to_join):
    new_list = []
    
    for sub_list_token in list_to_join:
        new_line = ','.join(sub_list_token)
        new_list.append(new_line)
    return new_list

new_total_list =  JoinSubLists(total_list)
new_ffmpeg_train_list = JoinSubLists(ffmpeg_train_list)
new_ffmpeg_test_list = JoinSubLists(ffmpeg_test_list)
#tokenizer = LoadSavedData(working_dir + 'common_tokenizer.pickle') # Baseline 2
tokenizer = LoadSavedData('D:\\Phd\\Backup\\2018-10-16-TIFS\\all_tokenizer_no_comments.pickle') # Baseline 1

total_sequences = tokenizer.texts_to_sequences(new_total_list)
ffmpeg_train_sequences = tokenizer.texts_to_sequences(new_ffmpeg_train_list)
ffmpeg_test_sequences = tokenizer.texts_to_sequences(new_ffmpeg_test_list)
word_index = tokenizer.word_index
print ('Found %s unique tokens.' % len(word_index))

print ("The length of tokenized sequence: " + str(len(total_sequences)))

# 2.2 Load the trained word2vec model.

#w2v_model_path = working_dir + 'CWE_w2v_model_CBOW.txt'# Baseline 2
w2v_model_path = 'D:\\Phd\\Backup\\2018-10-16-TIFS\\all_w2v_model_CBOW_no_comments.txt'# Baseline 1

w2v_model = open(w2v_model_path)

print ("----------------------------------------")
print ("The trained word2vec model: ")
print (w2v_model)

#------------------------------------#
# Plot tokenized sequences.
length_list = []

for item in total_sequences + ffmpeg_train_sequences + ffmpeg_test_sequences:
    length_list.append(len(item))

plt.hist(length_list, bins=30)
#plt.ylabel('Tokenized Sequences');
plt.show()

#------------------------------------#
# 3. Do the paddings.
#--------------------------------------------------------

print ("max_len ", MAX_LEN)
print('Pad sequences (samples x time)')

total_sequences_pad = pad_sequences(total_sequences, maxlen = MAX_LEN, padding ='post')
ffmpeg_train_sequence_pad = pad_sequences(ffmpeg_train_sequences, maxlen = MAX_LEN, padding ='post')
ffmpeg_test_sequence_pad = pad_sequences(ffmpeg_test_sequences, maxlen = MAX_LEN, padding ='post')

print ("The shape after paddings: ")
print (total_sequences_pad.shape)

print ("The shape after paddings: ")
print (ffmpeg_train_sequence_pad.shape)

print ("The shape after paddings: ")
print (ffmpeg_test_sequence_pad.shape)

#-------------------------------------------------------------#
# Use part of the ffmpeg training set as the actual training set, and the remaining set as the validation set.  

ffmpeg_train_set_x, ffmpeg_validation_set_x, ffmpeg_train_set_y, ffmpeg_validation_set_y, ffmpeg_train_set_id, ffmpeg_validation_set_id = train_test_split(ffmpeg_train_sequence_pad, ffmpeg_train_label, ffmpeg_train_list_id, test_size=0.7, random_state=42)
#
#train_set_x, validation_set_x, train_set_y, validation_set_y, train_set_id, validation_set_id = train_test_split(train_set_x, train_set_y, train_set_id, test_size=0.3, random_state=42)

train_set_x = total_sequences_pad.tolist() + ffmpeg_train_set_x.tolist()
train_set_x = np.asarray(train_set_x)
validation_set_x = ffmpeg_validation_set_x 
test_set_x = ffmpeg_test_sequence_pad

print ("Training set: ")

print (train_set_x)

print ("Validation set: ")

print (validation_set_x)

print ("Test set: ")

print (test_set_x)

train_set_y = np.asarray(total_list_label + ffmpeg_train_set_y.tolist())
validation_set_y = np.asarray(ffmpeg_validation_set_y)
test_set_y = np.asarray(ffmpeg_test_label)

print (len(train_set_x), len(train_set_y), len(total_list_id + ffmpeg_train_set_id), len(validation_set_y), len(ffmpeg_validation_set_id), len(validation_set_x), len(test_set_x), len(test_set_y), len(ffmpeg_test_list_id))

print (train_set_x.shape, train_set_y.shape, validation_set_x.shape, validation_set_y.shape, test_set_x.shape, test_set_y.shape)

print (np.count_nonzero(train_set_y), np.count_nonzero(validation_set_y), np.count_nonzero(test_set_y))

# -----------------------------------
# 4. Preparing the Embedding layer

embeddings_index = {} # a dictionary with mapping of a word i.e. 'int' and its corresponding 100 dimension embedding.

# Use the loaded model
for line in w2v_model:
   if not line.isspace():
       values = line.split()
       word = values[0]
       coefs = np.asarray(values[1:], dtype='float32')
       embeddings_index[word] = coefs
w2v_model.close()

print('Found %s word vectors.' % len(embeddings_index))

#embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM)) # The shape of embedding matrix (137, 100) because there are 137 common tokens.
#for word, i in word_index.items():
#   embedding_vector = embeddings_index.get(word) # the shape of the embedding_vector (100, )
#   if embedding_vector is not None:
#       # words not found in embedding index will be all-zeros.
#       embedding_matrix[i] = embedding_vector

# ====================================
# Get common tokens label       
# ====================================
#common_tokens = LoadSavedData(working_dir + 'common_token_list.pkl')

common_tokens = construc_list() 
#
embedding_matrix_1 = np.zeros((len(word_index) + 1, EMBEDDING_DIM + 1)) # The shape of embedding matrix (137, 100) because there are 137 common tokens.
for word, i in word_index.items():
   embedding_vector = embeddings_index.get(word) # the shape of the embedding_vector (100, )
   if word in common_tokens:
       common_token_label = 1
   else:
       common_token_label = 0
   if embedding_vector is not None:
       embedding_vector_list = embedding_vector.tolist()
       embedding_vector_list.append(common_token_label)
       embedding_vector_arr = np.asarray(embedding_vector_list)
       # words not found in embedding index will be all-zeros.
       embedding_matrix_1[i] = embedding_vector_arr

class_weight = {0:1, 1:8}

def train(train_set_x, train_set_y, validation_set_x, validation_set_y, saved_model_name):
    
    model = BiLSTM_network(MAX_LEN, EMBEDDING_DIM + 1, word_index, embedding_matrix_1, True)
    
    callbacks_list = [
       ModelCheckpoint(filepath = model_saved_path + os.sep + saved_model_name +'_{epoch:02d}_{val_acc:.3f}_{val_loss:3f}.h5', monitor='val_loss', verbose=1, save_best_only=True, period=1),
       EarlyStopping(monitor='val_loss', patience=PATIENCE, verbose=1, mode="min"),
		TensorBoard(log_dir=log_path, batch_size = BATCH_SIZE,  write_graph=True, write_grads=True, write_images=True, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None),
       CSVLogger(log_path + os.sep + saved_model_name + datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + '.log')]
    
    train_history = model.fit(train_set_x, train_set_y,
         epochs=EPOCHS,
         batch_size=BATCH_SIZE,
		   shuffle = False, # The data has already been shuffle before, so it is unnessary to shuffle it again. (And also, we need to correspond the ids to the features of the samples.)
         #validation_split=0.5,
         validation_data = (validation_set_x, validation_set_y), # Validation data is not used for training (or development of the model)
         callbacks=callbacks_list, # Get the best weights of the model and stop the first raound training.
         verbose=1,
         class_weight = class_weight)
    
    model.summary()
    
    return model, train_history

def test(test_set_x, test_set_y, model):
    
    #parallel_model = multi_gpu_model(model, gpus=2)
    
    model.compile(loss=LOSS_FUNCTION,
             optimizer=OPTIMIZER,
             metrics=['accuracy'])
    
    #model.summary()
    
    probs = model.predict(test_set_x, batch_size = BATCH_SIZE, verbose=1)
    
    predicted_classes = []

    for item in probs:
        if item[0] > 0.5:
            predicted_classes.append(1)
        else:
            predicted_classes.append(0)
    
    test_accuracy = np.mean(np.equal(test_set_y, predicted_classes))
    
    test_set_y = np.asarray(test_set_y)
    
    print ("LSTM classification result: ")
    print ("The accuracy: " + str(test_accuracy))
    
    target_names = ["Non-vulnerable","Vulnerable"] #non-vulnerable->0, vulnerable->1
    print (confusion_matrix(test_set_y, predicted_classes, labels=[0,1]))   
    print ("\r\n")
    print ("\r\n")
    print (classification_report(test_set_y, predicted_classes, target_names=target_names))
    
    return probs, test_accuracy
    
#def train_multip_inputs():
    
#def test_multip_inputs():
    

def plot_history(network_history):
    plt.figure()
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.plot(network_history.history['loss'])
    plt.plot(network_history.history['val_loss'])
    plt.legend(['Training', 'Validation'])
    plt.savefig('Epoch_loss_class_weight.jpg') 
    
#def printAttentionWords(test_set_x, test_set_y, model, word_index, num_word_pay_atten):
#    
#    vul_index = []
#    
#    for index, item in enumerate(test_set_y):
#        if item == 1:
#            vul_index.append(index)
#    
#    # Only visualize the vulnerable sequences. 
#    for item in vul_index:
#        visualize_attention(test_set_x, item, model, word_index, num_word_pay_atten)

if __name__ == '__main__':
    
    os.environ["CUDA_VISIBLE_DEVICES"]="0" # 1: Use the GTX 1070 only; 0: Use Titan XP only.

    model, train_history = train(train_set_x, train_set_y, validation_set_x, validation_set_y, 'BiLSTM_MA_part_of_ffmpeg_validation_class_weight_1')
    plot_history(train_history)
    
    best_model = load_model(working_dir + '//models//BiLSTM_MA_part_of_ffmpeg_validation_class_weight_1_160_0.964_0.148549.h5')
    
    # The attention model is a customized model, it is needed to use a dictionary to specify it.
    #best_model = load_model(working_dir + 'models//BiLSTM_baseline_attention_1_32_0.964_0.100730.h5', {'AttentionWithContext': AttentionWithContext})
    
    #probs, test_accuracy = test(test_set_x, test_set_y, best_model)
    probs, test_accuracy = test(validation_set_x, validation_set_y, best_model)
    
    #printAttentionWords(test_set_x, test_set_y, best_model, word_index, 20)
    
    ListToCSV(probs.tolist(), working_dir + '//prob_nn_part_of_ffmpeg_validation.csv')
    ListToCSV(ffmpeg_test_label, working_dir + '//prob_nn_label.csv')
    ListToCSV(ffmpeg_test_list_id, working_dir + '//prob_nn_id_new_embedding.csv')
    #K.clear_session()

#-------------------------------------------------------------------------#
# Get train and test set.
    
#working_dir_1 = 'D:\\Phd\\Backup\\2018-08-30-AsiaCCS\\manual_attention\\ffmpeg\\'    
#
#from os import listdir
#from os.path import isfile, join
#import shutil
#
#mypath = working_dir_1 + 'total_files\\'
#file_list = [f for f in listdir(mypath) if isfile(join(mypath, f))]
#
#for item in file_list:
#    if item in test_set_id:
#        shutil.copyfile(mypath + item + os.sep, working_dir_1 + 'Test\\' + item)
#    else:
#        shutil.copyfile(mypath + item + os.sep, working_dir_1 + 'Train\\' + item)
#        
"""
The FFmpeg data training set was completely used as validation set. 

[[3892    0]
 [ 145    0]]

                precision    recall  f1-score   support

Non-vulnerable       0.96      1.00      0.98      3892
    Vulnerable       0.00      0.00      0.00       145

     micro avg       0.96      0.96      0.96      4037
     macro avg       0.48      0.50      0.49      4037
  weighted avg       0.93      0.96      0.95      4037


#------------------------------------------------------------#
Part of the FFmpeg training set was used as actual training set, and the remaining was used as validation set.

[[3892    0]
 [ 145    0]]

                precision    recall  f1-score   support

Non-vulnerable       0.96      1.00      0.98      3892
    Vulnerable       0.00      0.00      0.00       145

     micro avg       0.96      0.96      0.96      4037
     macro avg       0.48      0.50      0.49      4037
  weighted avg       0.93      0.96      0.95      4037
  
"""
