# -*- coding: utf-8 -*-
"""
Created on Sat Feb  5 21:53:13 2022

@author: taha
"""

import numpy as np
import matplotlib.pyplot as plt
from IPython.core.debugger import set_trace
import pandas as pd 
from sklearn import datasets
import math 
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_classif
import statistics
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
from six.moves import urllib
import zipfile
from scipy import stats
import scipy.stats as st
import statsmodels.api as sm
from scipy.stats._continuous_distns import _distn_names
import matplotlib.pyplot as plt
from IPython.core.debugger import set_trace 
from statistics import mode
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from math import sqrt
from scipy.stats import mode
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE
from sklearn.linear_model import RidgeCV, LassoCV, Ridge, Lasso
from pandas import read_csv
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from IPython.core.debugger import set_trace

#Task1
#Aquire, preprocess and analyze the data
#1-describe the data set related to DR data with 19 attributes(x)and 155 instances
dataset_1= pd.read_csv ("http://archive.ics.uci.edu/ml/machine-learning-databases/hepatitis/hepatitis.data", names = ["Class", "AGE", "SEX", "STEROID", "ANTIVIRALS", "FATIGUE", "MALAISE", "ANOREXIA", "LIVER BIG", "LIVER FIRM", "SPLEEN PALPABLE", "SPIDERS", "ASCITES", "VARICES", "BILIRUBIN", "ALK PHOSPHATE", "SGOT", "ALBUMIN", "PROTIME", "HISTOLOGY"])
dataset_1.head()
print (dataset_1)


#Cleaning dataframe
#a-identify missing values in the Dataframe
dataset_delete=dataset_1[dataset_1.eq('?').any(1)]
#delete missing features from main Dataframe
dataset_1_Final=dataset_1[~dataset_1.isin(dataset_delete)].dropna()
print(dataset_1_Final)
help(dataset_1)


#3-using simple statistics
#a-converting columns to string and numerical
dataset_1_Final["BILIRUBIN"] = pd.to_numeric(dataset_1_Final["BILIRUBIN"]) 
dataset_1_Final["AGE"] = pd.to_numeric(dataset_1_Final["AGE"])
dataset_1_Final["ALK PHOSPHATE"] = pd.to_numeric(dataset_1_Final["ALK PHOSPHATE"]) 
dataset_1_Final["SGOT"] = pd.to_numeric(dataset_1_Final["SGOT"])
dataset_1_Final["ALBUMIN"] = pd.to_numeric(dataset_1_Final["ALBUMIN"])
dataset_1_Final["PROTIME"] = pd.to_numeric(dataset_1_Final["PROTIME"])
dataset_1_Final["SEX"]= pd.to_numeric(dataset_1_Final["SEX"])
dataset_1_Final["STEROID"]= pd.to_numeric(dataset_1_Final["STEROID"])
dataset_1_Final["ANTIVIRALS"]= pd.to_numeric(dataset_1_Final["ANTIVIRALS"])
dataset_1_Final["FATIGUE"]= pd.to_numeric(dataset_1_Final["FATIGUE"])
dataset_1_Final["MALAISE"]= pd.to_numeric(dataset_1_Final["MALAISE"])
dataset_1_Final["ANOREXIA"]= pd.to_numeric(dataset_1_Final["ANOREXIA"])
dataset_1_Final["LIVER BIG"]= pd.to_numeric(dataset_1_Final["LIVER BIG"])
dataset_1_Final["LIVER FIRM"]= pd.to_numeric(dataset_1_Final["LIVER FIRM"])
dataset_1_Final["SPLEEN PALPABLE"]= pd.to_numeric(dataset_1_Final["SPLEEN PALPABLE"])
dataset_1_Final["SPIDERS"]= pd.to_numeric(dataset_1_Final["SPIDERS"])
dataset_1_Final["ASCITES"]= pd.to_numeric(dataset_1_Final["ASCITES"])
dataset_1_Final["VARICES"]= pd.to_numeric(dataset_1_Final["VARICES"])
dataset_1_Final["HISTOLOGY"]= pd.to_numeric(dataset_1_Final["HISTOLOGY"])


#b-Specify features and class, find the distributions of the features and classes
x,y=dataset_1_Final[["AGE", "SEX", "STEROID", "ANTIVIRALS", "FATIGUE", "MALAISE", "ANOREXIA", "LIVER BIG", "LIVER FIRM", "SPLEEN PALPABLE", "SPIDERS", "ASCITES", "VARICES", "BILIRUBIN", "ALK PHOSPHATE", "SGOT", "ALBUMIN", "PROTIME", "HISTOLOGY"]],dataset_1_Final["Class"]   


x_2f=x[["ALBUMIN","PROTIME"]]
(N,D), C = x_2f.shape,int( np.max(y))                                              
print(f'instances (N) \t {N} \n features (D) \t {D} \n classes (C) \t {C}')
   
#c-Specify training data and testing data                                          
training_data = dataset_1_Final.sample(frac=0.8, random_state=25)
testing_data = dataset_1_Final.drop(training_data.index)


#d-separating features and class in train and test data
x_train_1,y_train_1=training_data[["AGE", "SEX", "STEROID", "ANTIVIRALS", "FATIGUE", "MALAISE", "ANOREXIA", "LIVER BIG", "LIVER FIRM", "SPLEEN PALPABLE", "SPIDERS", "ASCITES", "VARICES", "BILIRUBIN", "ALK PHOSPHATE", "SGOT", "ALBUMIN", "PROTIME", "HISTOLOGY"]],training_data[["Class"]]
x_test_1,y_test_1=testing_data[["AGE", "SEX", "STEROID", "ANTIVIRALS", "FATIGUE", "MALAISE", "ANOREXIA", "LIVER BIG", "LIVER FIRM", "SPLEEN PALPABLE", "SPIDERS", "ASCITES", "VARICES", "BILIRUBIN", "ALK PHOSPHATE", "SGOT", "ALBUMIN", "PROTIME", "HISTOLOGY"]],testing_data[["Class"]]


#e-seprating categorical features from numerical features
x_train_cat,x_train_numeric=training_data[["SEX", "STEROID", "ANTIVIRALS", "FATIGUE", "MALAISE", "ANOREXIA", "LIVER BIG", "LIVER FIRM", "SPLEEN PALPABLE", "SPIDERS", "ASCITES", "VARICES","HISTOLOGY"]],training_data[["AGE","BILIRUBIN","ALK PHOSPHATE","SGOT","ALBUMIN","PROTIME"]]
x_test_cat,x_test_numeric=testing_data[["SEX","STEROID", "ANTIVIRALS", "FATIGUE", "MALAISE", "ANOREXIA", "LIVER BIG", "LIVER FIRM", "SPLEEN PALPABLE", "SPIDERS", "ASCITES", "VARICES","HISTOLOGY"]],testing_data[["AGE","BILIRUBIN","ALK PHOSPHATE","SGOT","ALBUMIN","PROTIME"]]


x_train_2f = x_train_1[["ALBUMIN","PROTIME"]]
x_test_2f = x_test_1[["ALBUMIN","PROTIME"]]
x_2f=x[["ALBUMIN","PROTIME"]]

x_test=x_test_2f.to_numpy()
x_np=x_2f.to_numpy()
y_np=y.to_numpy()
x_train=x_train_2f.to_numpy()
yf_train=y_train_1.to_numpy()
yf_test=y_test_1.to_numpy()
y_train=yf_train.flatten()
y_test=yf_test.flatten()
y_train=y_train.astype('int')
y_test=y_test.astype('int')
y_np=y_np.astype('int')


#Task2
#Decision tree Algorithm
class Node:
    def __init__(self, data_indices, parent):
        self.data_indices = data_indices                    
        self.left = None                                     
        self.right = None                                   
        self.split_feature = None                           
        self.split_value = None                             
        if parent:
            self.depth = parent.depth + 1                   
            self.num_classes = parent.num_classes           
            self.data = parent.data                         
            self.labels = parent.labels                     
            class_prob = np.bincount(self.labels[data_indices], minlength=self.num_classes) 
            self.class_prob = class_prob / np.sum(class_prob)  
            
            
def greedy_test(node, cost_fn):
    
    best_cost = np.inf
    best_feature, best_value = None, None
    num_instances, num_features = node.data.shape
    
    data_sorted = np.sort(node.data[node.data_indices],axis=0)
    test_candidates = (data_sorted[1:] + data_sorted[:-1]) / 2.
    for f in range(num_features):
        
        data_f = node.data[node.data_indices, f]
        for test in test_candidates[:,f]:
            
            left_indices = node.data_indices[data_f <= test]
            right_indices = node.data_indices[data_f > test]
            
            
            if len(left_indices) == 0 or len(right_indices) == 0:                
                continue
                                                                 
            left_cost = cost_fn(node.labels[left_indices])
            right_cost = cost_fn(node.labels[right_indices])
            num_left, num_right = left_indices.shape[0], right_indices.shape[0]
            
            cost = (num_left * left_cost + num_right * right_cost)/num_instances
            
            if cost < best_cost:
                best_cost = cost
                best_feature = f
                best_value = test
    return best_cost, best_feature, best_value           

    
def cost_misclassification(labels):
    counts = np.bincount(labels) 
    class_probs = counts / np.sum(counts)
    
    return 1 - np.max(class_probs)


def cost_entropy(labels):
    class_probs = np.bincount(labels) / len(labels)
    class_probs = class_probs[class_probs > 0]              
    return -np.sum(class_probs * np.log(class_probs))       


def cost_gini_index(labels):
    class_probs = np.bincount(labels) / len(labels)
    return 1 - np.sum(np.square(class_probs))               


class DecisionTree:
    def __init__(self, num_classes=None, max_depth=3, cost_fn=cost_misclassification, min_leaf_instances=1):
        self.max_depth = max_depth       
        self.root = None                 
        self.cost_fn = cost_fn          
        self.num_classes = num_classes  
        self.min_leaf_instances = min_leaf_instances  
        
    def fit(self, data, labels):
        pass                            
    
    def predict(self, data_test):
        pass
    
    
def fit(self, data, labels):
    self.data = data
    self.labels = labels
    if self.num_classes is None:
        self.num_classes = np.max(labels) + 1
    
    self.root = Node(np.arange(data.shape[0]), None)
    self.root.data = data
    self.root.labels = labels
    self.root.num_classes = self.num_classes
    self.root.depth = 0
    
    self._fit_tree(self.root)
    return self

def _fit_tree(self, node):
    
    if node.depth == self.max_depth or len(node.data_indices) <= self.min_leaf_instances:
        return
    
    cost, split_feature, split_value = greedy_test(node, self.cost_fn)
    
    if np.isinf(cost):
        return
    
    
    test = node.data[node.data_indices,split_feature] <= split_value
    
    node.split_feature = split_feature
    node.split_value = split_value
    
    left = Node(node.data_indices[test], node)
    right = Node(node.data_indices[np.logical_not(test)], node)
    
    self._fit_tree(left)
    self._fit_tree(right)
    
    node.left = left
    node.right = right

DecisionTree.fit = fit
DecisionTree._fit_tree = _fit_tree


def predict(self, data_test):
    class_probs = np.zeros((data_test.shape[0], self.num_classes))
    for n, x in enumerate(data_test):
        node = self.root
        
        while node.left:
            if x[node.split_feature] <= node.split_value:
                node = node.left
            else:
                node = node.right
        
        class_probs[n,:] = node.class_prob
    return class_probs

DecisionTree.predict = predict


(num_instances, num_features), num_classes = x.shape,int(np.max(y))


#Task3
#1-Testing Accuracy
tree = DecisionTree(max_depth=1)
probs_test = tree.fit(x_train, y_train).predict(x_test)
y_pred = np.argmax(probs_test,1)
accuracy = np.sum(y_pred == y_test)/y_test.shape[0]
print(f'Testing accuracy is {accuracy*100:.1f}.')



#2-Training Accuracy
tree = DecisionTree(max_depth=1)
probs_test = tree.fit(x_train, y_train).predict(x_train)
y_pred = np.argmax(probs_test,1)
accuracy = np.sum(y_pred == y_train)/y_train.shape[0]
print(f'Training accuracy is {accuracy*100:.1f}.')



#3-Vlidation Accuracy
S1 = training_data.sample(frac=0.25, random_state=25)
X1=training_data.drop(S1.index)
S12,yS12=S1[[ "ALBUMIN", "PROTIME"]],S1[["Class"]]
X12,yX12=X1[["ALBUMIN", "PROTIME"]],X1[["Class"]]
X12np=X12.to_numpy()
S12np=S12.to_numpy()
yX12np=yX12.to_numpy()
yS12np=yS12.to_numpy()
yS12npf=yS12np.flatten()
yX12npf=yX12np.flatten()
yS12npf=yS12npf.astype('int')
yX12npf=yS12npf.astype('int')
tree = DecisionTree(max_depth=1)
probs_test = tree.fit(S12np, yX12npf).predict(S12np)
y_pred = np.argmax(probs_test,1)
accuracy1 = np.sum(y_pred == yS12npf)/yS12npf.shape[0]


S2 = X1.sample(frac=0.33, random_state=25)
X3 = X1.drop(S2.index)
X2 = X3 + S1
S22,yS22=S2[[ "ALBUMIN", "PROTIME"]],S2[["Class"]]
X22,yX22=X2[["ALBUMIN", "PROTIME"]],X2[["Class"]]
X22np=X22.to_numpy()
S22np=S22.to_numpy()
yX22np=yX22.to_numpy()
yS22np=yS22.to_numpy()
yS22npf=yS22np.flatten()
yX22npf=yX22np.flatten()
yS22npf=yS22npf.astype('int')
yX22npf=yS22npf.astype('int')
tree = DecisionTree(max_depth=1)
probs_test = tree.fit(S22np, yX22npf).predict(S22np)
y_pred = np.argmax(probs_test,1)
accuracy2 = np.sum(y_pred == yS22npf)/yS22npf.shape[0]


S4 = X3.sample(frac=0.5, random_state=25)
X5 = X3.drop(S4.index)
X4 = X5 + S1 + S2
S42,yS42=S4[[ "ALBUMIN", "PROTIME"]],S4[["Class"]]
X42,yX42=X4[["ALBUMIN", "PROTIME"]],X4[["Class"]]
X42np=X42.to_numpy()
S42np=S42.to_numpy()
yX42np=yX42.to_numpy()
yS42np=yS42.to_numpy()
yS42npf=yS42np.flatten()
yX42npf=yX42np.flatten()
yS42npf=yS42npf.astype('int')
yX42npf=yS42npf.astype('int')
tree = DecisionTree(max_depth=1)
probs_test = tree.fit(S42np, yX42npf).predict(S42np)
y_pred = np.argmax(probs_test,1)
accuracy3 = np.sum(y_pred == yS42npf)/yS42npf.shape[0]


S6 = X5
X6 = S1 + S2 + S4
S62,yS62=S6[[ "ALBUMIN", "PROTIME"]],S6[["Class"]]
X62,yX62=X6[["ALBUMIN", "PROTIME"]],X6[["Class"]]
X62np=X62.to_numpy()
S62np=S62.to_numpy()
yX62np=yX62.to_numpy()
yS62np=yS62.to_numpy()
yS62npf=yS62np.flatten()
yX62npf=yX62np.flatten()
yS62npf=yS62npf.astype('int')
yX62npf=yS62npf.astype('int')
tree = DecisionTree(max_depth=1)
probs_test = tree.fit(S62np, yX62npf).predict(S62np)
y_pred = np.argmax(probs_test,1)
accuracy4 = np.sum(y_pred == yS62npf)/yS62npf.shape[0]

average_Accuracy=((accuracy1+accuracy2+accuracy3+accuracy4)*100)/4
print('Validation accuracy=', average_Accuracy)



#4-Decision boundry
correct = y_test == y_pred
incorrect = np.logical_not(correct)
plt.scatter(x_train[:,0], x_train[:,1], c=y_train, marker='o', alpha=.2, label='train')
plt.scatter(x_test[correct,0], x_test[correct,1], marker='.', c=y_pred[correct], label='correct')
plt.scatter(x_test[incorrect,0], x_test[incorrect,1], marker='x', c=y_test[incorrect], label='misclassified')
plt.legend()
plt.show()

x0v = np.linspace(np.min(x_np[:,0]), np.max(x_np[:,0]), 200)
x1v = np.linspace(np.min(x_np[:,1]), np.max(x_np[:,1]), 200)
x0,x1 = np.meshgrid(x0v, x1v)
x_all = np.vstack((x0.ravel(),x1.ravel())).T

model = DecisionTree(max_depth=1)
y_train_prob = np.zeros((y_train.shape[0], (num_classes)+1))
y_train_prob[np.arange(y_train.shape[0]), y_train] = 1
y_prob_all = model.fit(x_train, y_train).predict(x_all)
plt.scatter(x_train[:,0], x_train[:,1], c=y_train_prob, marker='o', alpha=1)
plt.scatter(x_all[:,0], x_all[:,1], c=y_prob_all, marker='.', alpha=.01)
plt.ylabel('ALBUMIN')
plt.xlabel('PROTIME')
plt.show()





























