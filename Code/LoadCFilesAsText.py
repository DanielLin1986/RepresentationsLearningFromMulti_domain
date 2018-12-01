# -*- coding: utf-8 -*-
"""
Created on Wed Apr 11 11:32:26 2018

@author: yuyu-

Load C files as text. Separate the token '*var', so that a space is added to '*' and 'var' to form two tokens.

"""
import pickle
import numpy as np
import csv
import os
import matplotlib.pyplot as plt
import pandas as pd
#import plotly.plotly as py

#test_str = 'mian(int x) {int x= 1 +2}'

# Separate '(', ')', '{', '}', '*', '/', '+', '-', '=', ';', '[', ']' characters.
def SplitCharacters(str_to_split):
    #Character_sets = ['(', ')', '{', '}', '*', '/', '+', '-', '=', ';', ',']
    str_list_str = ''
    
    if '(' in str_to_split:
        str_to_split = str_to_split.replace('(', ' ( ') # Add the space before and after the '(', so that it can be split by space.
        str_list = str_to_split.split(' ')
        str_list_str = ' '.join(str_list)
    
    if ')' in str_to_split:
        str_to_split = str_to_split.replace(')', ' ) ') # Add the space before and after the ')', so that it can be split by space.
        str_list = str_to_split.split(' ')
        str_list_str = ' '.join(str_list)
        
    if '{' in str_to_split:
        str_to_split = str_to_split.replace('{', ' { ')
        str_list = str_to_split.split(' ')
        str_list_str = ' '.join(str_list)
    
    if '}' in str_to_split:
        str_to_split = str_to_split.replace('}', ' } ')
        str_list = str_to_split.split(' ')
        str_list_str = ' '.join(str_list)
    
    if '*' in str_to_split:
        str_to_split = str_to_split.replace('*', ' * ')
        str_list = str_to_split.split(' ')
        str_list_str = ' '.join(str_list)
    
    if '/' in str_to_split:
        str_to_split = str_to_split.replace('/', ' / ')
        str_list = str_to_split.split(' ')
        str_list_str = ' '.join(str_list)
        
    if '+' in str_to_split:
        str_to_split = str_to_split.replace('+', ' + ')
        str_list = str_to_split.split(' ')
        str_list_str = ' '.join(str_list)
    
    if '-' in str_to_split:
        str_to_split = str_to_split.replace('-', ' - ')
        str_list = str_to_split.split(' ')
        str_list_str = ' '.join(str_list)
        
    if '=' in str_to_split:
        str_to_split = str_to_split.replace('=', ' = ')
        str_list = str_to_split.split(' ')
        str_list_str = ' '.join(str_list)
    
    if ';' in str_to_split:
        str_to_split = str_to_split.replace(';', ' ; ')
        str_list = str_to_split.split(' ')
        str_list_str = ' '.join(str_list)
    
    if '[' in str_to_split:
        str_to_split = str_to_split.replace('[', ' [ ')
        str_list = str_to_split.split(' ')
        str_list_str = ' '.join(str_list)
    
    if ']' in str_to_split:
        str_to_split = str_to_split.replace(']', ' ] ')
        str_list = str_to_split.split(' ')
        str_list_str = ' '.join(str_list)
        
    if '>' in str_to_split:
        str_to_split = str_to_split.replace('>', ' > ')
        str_list = str_to_split.split(' ')
        str_list_str = ' '.join(str_list)
        
    if '<' in str_to_split:
        str_to_split = str_to_split.replace('<', ' < ')
        str_list = str_to_split.split(' ')
        str_list_str = ' '.join(str_list)
    
    if '"' in str_to_split:
        str_to_split = str_to_split.replace('"', ' " ')
        str_list = str_to_split.split(' ')
        str_list_str = ' '.join(str_list)
        
    if '->' in str_to_split:
        str_to_split = str_to_split.replace('->', ' -> ')
        str_list = str_to_split.split(' ')
        str_list_str = ' '.join(str_list)
    
    if '>>' in str_to_split:
        str_to_split = str_to_split.replace('>>', ' >> ')
        str_list = str_to_split.split(' ')
        str_list_str = ' '.join(str_list)
    
    if '<<' in str_to_split:
        str_to_split = str_to_split.replace('<<', ' << ')
        str_list = str_to_split.split(' ')
        str_list_str = ' '.join(str_list)
    
    if ',' in str_to_split:
        str_to_split = str_to_split.replace(',', ' , ')
        str_list = str_to_split.split(' ')
        str_list_str = ' '.join(str_list)
        
    if str_list_str is not '':
        return str_list_str
    else:
        return str_to_split

#test_new_char = SplitCharacters(test_str)

#print (test_new_char)
        
def SavedData(path, file_to_save):
    with open(path, 'wb') as handle:
        #pickle.dump(file_to_save, handle, protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(file_to_save, handle, protocol=2)

def Save3DList(save_path, list_to_save):
    with open(save_path, 'w', encoding='latin1') as f:
        wr = csv.writer(f, quoting=csv.QUOTE_ALL)
        wr.writerows(list_to_save)
        

def Save2DList(save_path, list_to_save):
    with open(save_path, 'w', encoding='latin1') as f:
        wr = csv.writer(f, quoting=csv.QUOTE_ALL)
        wr.writerow(list_to_save)
        
def ListToCSV(list_to_csv, path):
   df = pd.DataFrame(list_to_csv)
   df.to_csv(path, index=False)
   
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

# Further split the elements such as "const int *" into "const", "int" and "*"
def ProcessList(list_to_process):
    token_list = []
    for sub_list_to_process in list_to_process:
        sub_token_list = []
        if len(sub_list_to_process) != 0:
            for each_word in sub_list_to_process: # Remove the empty row
                each_word = str(each_word)
                sub_word = each_word.split()
                for element in sub_word:
                    sub_token_list.append(element)
            token_list.append(sub_token_list)
    return token_list

# 
def getCFilesFromText(path):
    files_list = []
    file_id_list = []
    for fpathe,dirs,fs in os.walk(path):
        for f in fs:
            if (os.path.splitext(f)[1] == '.c'):
                file_id_list.append(f)  
            if (os.path.splitext(f)[1] == '.c'):
                with open(path + f, encoding='latin1') as file:
                    lines = file.readlines()
                    file_list = []
                    for line in lines:
                        if line is not ' ' and line is not '\n': # Remove sapce and line-change characters
                            sub_line = line.split()
                            new_sub_line = []
                            for element in sub_line:
                                new_element = SplitCharacters(element)
                                new_sub_line.append(new_element)
                            new_line = ' '.join(new_sub_line)
                            file_list.append(new_line)
                    new_file_list = ' '.join(file_list)
                    split_by_space = new_file_list.split()
                files_list.append(split_by_space)
            #files_list = removeSemicolon(files_list)
            #files_list = ProcessList(files_list)
        return files_list, file_id_list

# The Vulnerable and Non-vulnerable files.
#non_vul_file_list, non_vul_file_list_id = getCFilesFromText(working_dir + 'non_vul_files\\')     
#vul_file_list, vul_file_list_id = getCFilesFromText(working_dir + 'vul_files\\')  
#
#new_non_vul_file_list = removeSemicolon(non_vul_file_list)
#new_vul_file_list = removeSemicolon(vul_file_list)
#
#print ("The length non-vulnerable files: " + str(len(new_non_vul_file_list)) + " . ID: " +  str(len(non_vul_file_list_id)))
#print ("The length vulnerable files: " + str(len(new_vul_file_list)) + " . ID: " +  str(len(vul_file_list_id)))
#
#print ("-------------------------------------------------------")
#
#non_vul_file_list_label = [0] * len(new_non_vul_file_list)
#
#vul_file_list_label = [1] * len(new_vul_file_list)
#
#print ("Total length non-vulnerable files: " + str(len(non_vul_file_list_label)) + " . ID: " +  str(len(non_vul_file_list_id)))
#print ("Total length vulnerable files: " + str(len(vul_file_list)) + " . ID: " +  str(len(vul_file_list_id)))
#
#total_list_all = new_non_vul_file_list + new_vul_file_list
#
#length_list = []
#
#for item in total_list_all:
#    length_list.append(len(item))
#
#plt.hist(length_list, 50)
#
#the_smallest = min(length_list)
#print (length_list.index(min(length_list)))
#the_longest = max(length_list)
#
#SavedData(working_dir + 'non_vul_list.pkl', new_non_vul_file_list)
#SavedData(working_dir + 'non_vul_list_id.pkl', non_vul_file_list_id) 
#SavedData(working_dir + 'non_vul_list_label.pkl', non_vul_file_list_label) 
#
#SavedData(working_dir + 'vul_list.pkl', new_vul_file_list)
#SavedData(working_dir + 'vul_list_id.pkl', vul_file_list_id) 
#SavedData(working_dir + 'vul_list_label.pkl', vul_file_list_label) 
#
#ListToCSV(new_non_vul_file_list, working_dir + 'non_vul_list.csv')
#ListToCSV(new_vul_file_list, working_dir + 'vul_list.csv')
#SavedData(working_dir + 'train_file_list.pkl', train_file_list)
#SavedData(working_dir + 'train_file_list_id.pkl', train_file_list_id)  
                   
#Save3DList(working_dir + 'train_files_list.csv', files_list)
#np.savetxt(working_dir + 'SampleData\\files_list.csv', files_list, delimiter=',')
#Save2DList(working_dir + 'train_files_list_id.csv', file_id_list)
