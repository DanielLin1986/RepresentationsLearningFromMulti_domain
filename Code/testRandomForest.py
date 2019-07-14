# -*- coding: utf-8 -*-
"""
Created on Tue Mar  6 12:06:27 2018

@author: yuyu-
"""

import time
import pickle
import numpy as np
from numpy import genfromtxt
import pandas as pd

from itertools import chain

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
#from sklearn.grid_search import GridSearchCV
from sklearn.model_selection import cross_val_predict
from sklearn.ensemble import RandomForestClassifier

script_start_time = time.time()

print ("Script starts at: " + str(script_start_time))

working_dir = 'D:\\Path\\to\\your\\data\\'

libtiff_train_repre_nist = np.loadtxt(open(working_dir + 'libtiff_train_rep_128.csv', 'rb'), delimiter=',', dtype='float')
libtiff_test_repre_nist= np.loadtxt(open(working_dir + 'libtiff_test_rep_128.csv', 'rb'), delimiter=',', dtype='float')

libtiff_train_label = np.loadtxt(open(working_dir + 'libtiff_train_rep_label.csv', 'rb'), delimiter=',', dtype='int')
libtiff_test_label = np.loadtxt(open(working_dir + 'libtiff_test_rep_label.csv', 'rb'), delimiter=',', dtype='int')

train_id = np.loadtxt(open(working_dir + 'libtiff_train_id.csv', 'rb'), delimiter=',', dtype='str').tolist()
test_id = np.loadtxt(open(working_dir + 'libtiff_test_id.csv', 'rb'), delimiter=',', dtype='str').tolist()

train_set_x = libtiff_train_repre_nist
train_set_y = libtiff_train_label

test_set_x = libtiff_test_repre_nist
test_set_y = libtiff_test_label

print ("-------------------------")

print ("The shape of the datasets: " + "\r\n")

print (len(train_set_x), len(train_set_y), len(test_set_x), len(test_set_y))

print (train_set_x.shape, train_set_y.shape, test_set_x.shape, test_set_y.shape)

print (np.count_nonzero(train_set_y), np.count_nonzero(test_set_y))

print ("The shape of training and testing sets are: ")

print (train_set_x.shape)
print (test_set_x.shape)

#-------------------------------------------------------------------------------
# Invoke Sklearn tools for classification -- using Random Forest

#train a random forest model
print ("Fitting the classifier to the training set") 
#t0 = time()

def invokeRandomForest(train_set_x, train_set_y, test_set_x, test_set_y):
#    param_grid = {'max_depth': [15,20,25,30],
#                  'min_samples_split': [4,5,6],
#                  'min_samples_leaf': [2,3,4,5],
#                  'bootstrap': [True,False],
#                  'criterion': ['gini','entropy'],
#                  'n_estimators': [40,50,55,60,65]}
#    
    train_set_y = np.ndarray.flatten(np.asarray(train_set_y))
    test_set_y = np.ndarray.flatten(np.asarray(test_set_y))
    
    #clf = GridSearchCV(RandomForestClassifier(), param_grid=param_grid, n_jobs=-1)
    clf = RandomForestClassifier(bootstrap=True, class_weight='balanced', #class_weight={0:1, 1:4},
            criterion='entropy', max_depth=40, max_features='auto',
            max_leaf_nodes=None, min_impurity_decrease=0.0,
            min_impurity_split=None, min_samples_leaf=3,
            min_samples_split=4, min_weight_fraction_leaf=0.0,
            n_estimators=8000, oob_score=False, random_state=None,
            verbose=1, warm_start=False, n_jobs=-1)
    clf = clf.fit(train_set_x, train_set_y)
    
    print("feature importance:")
    print(clf.feature_importances_)
    print ("\n")
    
    #print("best estimator found by grid search:")
    #print(clf.best_estimator_)
    
    print ("\r\n")
    
    #evaluate the model on the test set
    print("predicting on the test set")
    #t0 = time()
    y_predict = clf.predict(test_set_x)
    
    y_predict_proba = clf.predict_proba(test_set_x)
    
    # Accuracy
    accuracy = np.mean(test_set_y==y_predict)*100
    print ("accuracy = " +  str(accuracy))
        
    target_names = ["Non-defective","Defective"] #non-buggy->0, buggy->1
    print (confusion_matrix(test_set_y, y_predict, labels=[0,1]))   
    print ("\r\n")
    print ("\r\n")
    print (classification_report(test_set_y, y_predict, target_names=target_names))
    
    return y_predict_proba

if __name__ == '__main__':
    y_predict_proba = invokeRandomForest(train_set_x, train_set_y, test_set_x, test_set_y)

print ("\r\n")
print ("--- %s seconds ---" + str(time.time() - script_start_time))

np.savetxt(working_dir + 'result_sard_weighted_128.csv', y_predict_proba, delimiter=",")
