# -*- coding: utf-8 -*-
"""
Created on Sun Apr 15 12:20:53 2018

@author: yuyu-

Obtain representations from the network which was trained using CVE data.

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

with tf.device('/gpu:1'):
# ------------------------------------------------------------ #
# Parameters used and data directories

    LOSS_FUNCTION = 'binary_crossentropy'
    #OPTIMIZER = 'adamax'
    OPTIMIZER = 'adamax'
    
    project_name = 'LibTIFF'
    
    MAX_LEN = 1000 # The Padding Length for each sample.
    EMBEDDING_DIM = 100 # The Embedding Dimension for each element within the sequence of a data sample.
    
    working_dir = 'D:\\Phd\\Backup\\2018-10-16-TIFS\\' + project_name  + os.sep
    
    model_saved_path = 'D:\\Phd\\Backup\\2018-10-16-TIFS\\models\\'
    
    model_name = 'BiLSTM_sard_pretrained_all_tokens_97_1.000_0.000679.h5'
    
    sample_data_dir = 'D:\\Phd\\Backup\\2018-04-06-multi-source-data\\Experiment_04-13-2018\\SampleData\\'
    
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
            i = [x for x in i if str(x) != ';'] # Remove the ';' in the libtiff list.
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
    libtiff_test_list = LoadSavedData(working_dir + 'test_file_list.pkl')
    libtiff_test_list_id = LoadSavedData(working_dir + 'test_file_list_id.pkl')
    
    libtiff_train_list = LoadSavedData(working_dir + 'train_file_list.pkl')
    libtiff_train_list_id = LoadSavedData(working_dir + 'train_file_list_id.pkl')
    
    #libtiff_test_list, libtiff_test_list_id = getData(working_dir + 'libtiff_test.csv')
    libtiff_test_label = GenerateLabels(libtiff_test_list_id)
    #
    #libtiff_train_list, libtiff_train_list_id = getData(working_dir + 'libtiff_train.csv')
    libtiff_train_label = GenerateLabels(libtiff_train_list_id)
    
    libtiff_test_list = ProcessList_1(libtiff_test_list)
    libtiff_test_list = processList(libtiff_test_list)
    libtiff_train_list = ProcessList_1(libtiff_train_list)
    libtiff_train_list = processList(libtiff_train_list)
    
    total_list = libtiff_train_list + libtiff_test_list
    total_list_id = libtiff_train_list_id + libtiff_test_list_id
    total_list_label = libtiff_train_label.tolist() + libtiff_test_label.tolist()
    total_list_label = np.ndarray.flatten(np.asarray(total_list_label))
    
    libtiff_vul_list = []
    libtiff_non_vul_list = []
    libtiff_non_vul_list_id = []
    libtiff_vul_list_id = []
    
    #for index, item in enumerate(total_list_label):
    #    if item == 1:
    #        ffmpeg_vul_list.append(total_list[index])
    #        ffmpeg_vul_list_id.append(total_list_id[index])
    #    else:
    #        ffmpeg_non_vul_list.append(total_list[index])
    #        ffmpeg_non_vul_list_id.append(total_list_id[index])
    #
    #random_index =  randrange(100, len(ffmpeg_non_vul_list))
    #ffmpeg_non_vul_rdm = ffmpeg_non_vul_list[random_index-100:random_index]
    #ffmpeg_non_vul_rdm_id = ffmpeg_non_vul_list_id[random_index-100:random_index]
    #ffmpeg_non_vul_rdm_label = [2] * 100
    #
    #random_index =  randrange(40, len(ffmpeg_vul_list))
    #ffmpeg_vul_rdm = ffmpeg_vul_list[random_index-40:random_index]
    #ffmpeg_vul_rdm_id = ffmpeg_vul_list_id[random_index-100:random_index]
    #ffmpeg_vul_rdm_label = [3] * 40
    #
    ##------------------------------------------------------------------------------
    ## Get the Sample data.
    #        
    #cwe_non_vul = LoadSavedData(sample_data_dir + 'total_non_vul_list.pkl')
    #cwe_non_vul_label = LoadSavedData(sample_data_dir + 'total_non_vul_list_label.pkl')
    #cwe_non_vul_id = LoadSavedData(sample_data_dir + 'total_non_vul_list_id.pkl')
    #cwe_non_vul = processList(cwe_non_vul)
    #
    #cwe_vul = LoadSavedData(sample_data_dir + 'total_vul_list.pkl')
    #cwe_vul_label = LoadSavedData(sample_data_dir + 'total_vul_list_label.pkl')
    #cwe_vul_id = LoadSavedData(sample_data_dir + 'total_vul_list_id.pkl')
    #cwe_vul =  processList(cwe_vul)
    #
    #random_index =  randrange(200, len(cwe_non_vul))
    #cwe_non_vul_rdm = cwe_non_vul[random_index-200:random_index]
    #cwe_non_vul_rdm_id = cwe_non_vul_id[random_index-200:random_index]
    #cwe_non_vul_rdm_label = [0] * 200
    #
    #random_index =  randrange(200, len(cwe_vul))
    #cwe_vul_rdm = cwe_vul[random_index-200:random_index]
    #cwe_vul_rdm_id = cwe_vul_id[random_index-200:random_index]
    #cwe_vul_rdm_label = [1] * 200
    
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
    
    #------------------------------------#
    # 4. Load pre-trained word2vec and tokens
    
    print ("Applying tokenization....")
    
    def JoinSubLists(list_to_join):
        new_list = []
        
        for sub_list_token in list_to_join:
            new_line = ','.join(sub_list_token)
            new_list.append(new_line)
        return new_list
    
    libtiff_train_list_new = JoinSubLists(libtiff_train_list)
    libtiff_test_list_new = JoinSubLists(libtiff_test_list)
    
    #non_vul_list_new = JoinSubLists(cwe_non_vul_rdm)
    #vul_list_new = JoinSubLists(cwe_vul_rdm)
    
    tokenizer = LoadSavedData('D:\\Phd\\Backup\\2018-10-16-TIFS\\all_tokenizer_no_comments.pickle')
    libtiff_train_sequences = tokenizer.texts_to_sequences(libtiff_train_list_new)
    libtiff_test_sequences = tokenizer.texts_to_sequences(libtiff_test_list_new)
    
    #non_vul_sequences = tokenizer.texts_to_sequences(non_vul_list_new)
    #vul_sequences = tokenizer.texts_to_sequences(vul_list_new)
    
    word_index = tokenizer.word_index
    print ('Found %s unique tokens.' % len(word_index))
    
    print ("The length of tokenized training sequences: " + str(len(libtiff_train_sequences)))
    print ("The length of tokenized testing sequences: " + str(len(libtiff_test_sequences)))
    
    print ("Loading trained word2vec model...")
    
    # Load the pre-trained embeddings.
    w2v_model_path = 'D:\\Phd\\Backup\\2018-10-16-TIFS\\all_w2v_model_CBOW_no_comments.txt'
    w2v_model = open(w2v_model_path, encoding="latin1")
    
    print ("----------------------------------------")
    print ("The trained word2vec model: ")
    print (w2v_model)
    
    #------------------------------------#
    # 3. Do the paddings.
    print ("max_len ", MAX_LEN)
    print('Pad sequences (samples x time)')
    
    libtiff_train_sequences_pad = pad_sequences(libtiff_train_sequences, maxlen = MAX_LEN, padding ='post')
    libtiff_test_sequences_pad = pad_sequences(libtiff_test_sequences, maxlen = MAX_LEN, padding ='post')
    print (libtiff_train_sequences_pad.shape)
    print (libtiff_test_sequences_pad.shape)
    
    #non_vul_sequences_pad = pad_sequences(non_vul_sequences, maxlen = MAX_LEN, padding ='post')
    #vul_sequences_pad = pad_sequences(vul_sequences, maxlen = MAX_LEN, padding ='post')
    #print (non_vul_sequences_pad.shape)
    #print (vul_sequences_pad.shape)
    
    # Acquire the embeddings.
    def ObtainRepresentations(input_sequences, layer_number, model):
        layered_model = Model(inputs = model.input, outputs=model.layers[layer_number].output)
        representations = layered_model.predict(input_sequences)
        return representations
    
    # With 2 dropout layers，there are 10 layers in total. So we use the second last layer
    libtiff_train_repre_nist = ObtainRepresentations(libtiff_train_sequences_pad, 6, model) 
    libtiff_test_repre_nist = ObtainRepresentations(libtiff_test_sequences_pad, 6, model) 
    
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
    
    np.savetxt(working_dir + 'libtiff_train_nist_rep_128.csv', libtiff_train_repre_nist, delimiter=",")
    np.savetxt(working_dir + 'libtiff_train_nist_label.csv', libtiff_train_label, delimiter=",")
    np.savetxt(working_dir + 'libtiff_test_nist_rep_128.csv', libtiff_test_repre_nist, delimiter=",")
    np.savetxt(working_dir + 'libtiff_test_nist_label.csv', libtiff_test_label, delimiter=",")
    
    ListToCSV(libtiff_test_list_id, working_dir + 'libtiff_test_id.csv')
    ListToCSV(libtiff_train_list_id, working_dir + 'libtiff_train_id.csv')

#------------------------------------------------------------------------------
# Get the ids of the data that satisfy certain requirements
#set_1 = []
#
#for index, item in enumerate(result_set):
#    if item[0] < 9 and item[0] > 0 and item[1] > 40 and item [1] < 46:
#        set_1.append(total_list_id[index])