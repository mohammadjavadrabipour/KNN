 # -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

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


#2-cleaning dataframe 
#a_identify missing values in the Dataframe
dataset_delete=dataset_1[dataset_1.eq('?').any(1)]
#b_delete missing features from main Dataframe
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


#find the distributions of the features and classes
plt.hist(dataset_1_Final['AGE'], bins=50)
plt.show()
desc=dataset_1_Final.describe()
print(desc)
   
#b-Specify features and class, find the distributions of the features and classes
x,y=dataset_1_Final[["AGE", "SEX", "STEROID", "ANTIVIRALS", "FATIGUE", "MALAISE", "ANOREXIA", "LIVER BIG", "LIVER FIRM", "SPLEEN PALPABLE", "SPIDERS", "ASCITES", "VARICES", "BILIRUBIN", "ALK PHOSPHATE", "SGOT", "ALBUMIN", "PROTIME", "HISTOLOGY"]],dataset_1_Final["Class"]   


x_2f=x[["ALBUMIN","PROTIME"]]
(N,D), C = x_2f.shape,int( np.max(y))                                              
print(f'instances (N) \t {N} \n features (D) \t {D} \n classes (C) \t {C}')
            
                                 
#c-Specify training data and testing data 
training_data = dataset_1_Final.sample(frac=0.8, random_state=25)
testing_data = dataset_1_Final.drop(training_data.index)


#d-separating features and class in train and test data
x_train_1,y_train_1=training_data[["AGE", "SEX", "STEROID", "ANTIVIRALS", "FATIGUE", "MALAISE", "ANOREXIA", "LIVER BIG", "LIVER FIRM", "SPLEEN PALPABLE", "SPIDERS", "ASCITES", "VARICES", "BILIRUBIN", "ALK PHOSPHATE", "SGOT", "ALBUMIN", "PROTIME", "HISTOLOGY","Class"]],training_data[["Class"]]
x_test_1,y_test_1=testing_data[["AGE", "SEX", "STEROID", "ANTIVIRALS", "FATIGUE", "MALAISE", "ANOREXIA", "LIVER BIG", "LIVER FIRM", "SPLEEN PALPABLE", "SPIDERS", "ASCITES", "VARICES", "BILIRUBIN", "ALK PHOSPHATE", "SGOT", "ALBUMIN", "PROTIME", "HISTOLOGY"]],testing_data[["Class"]]

  
x_train_cat,x_train_numeric=training_data[["SEX", "STEROID", "ANTIVIRALS", "FATIGUE", "MALAISE", "ANOREXIA", "LIVER BIG", "LIVER FIRM", "SPLEEN PALPABLE", "SPIDERS", "ASCITES", "VARICES","HISTOLOGY"]],training_data[["AGE","BILIRUBIN","ALK PHOSPHATE","SGOT","ALBUMIN","PROTIME"]]
x_test_cat,x_test_numeric=testing_data[["SEX","STEROID", "ANTIVIRALS", "FATIGUE", "MALAISE", "ANOREXIA", "LIVER BIG", "LIVER FIRM", "SPLEEN PALPABLE", "SPIDERS", "ASCITES", "VARICES","HISTOLOGY"]],testing_data[["AGE","BILIRUBIN","ALK PHOSPHATE","SGOT","ALBUMIN","PROTIME"]]


#f-prepare input data
oe = OrdinalEncoder()
oe.fit(x_train_cat)
x_train_enc = oe.transform(x_train_cat)
x_test_enc = oe.transform(x_test_cat)

 
#g-prepare target
le = LabelEncoder()
le.fit(y_train_1)
y_train_enc = le.transform(y_train_1)
y_test_enc = le.transform(y_test_1)


#h-Using Pearson Correlation
plt.figure(figsize=(12,10))
cor = x_train_1.corr()
sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
plt.show()

print(x_train_1[["PROTIME","ALBUMIN"]].corr())
print(x_train_1[["ALBUMIN","ASCITES"]].corr())
print(x_train_1[["PROTIME","ASCITES"]].corr())
print(x_train_1[["PROTIME","HISTOLOGY"]].corr())
print(x_train_1[["ASCITES","HISTOLOGY"]].corr())
print(x_train_1[["ALBUMIN","HISTOLOGY"]].corr())

#i-visualization of the data
ax=training_data.plot(x='PROTIME',y='ALBUMIN',c='Class', cmap="viridis",marker="*", kind="scatter",s=50,label='train')
testing_data.plot(x='PROTIME',y='ALBUMIN',c='Class', cmap="viridis",marker="o", kind="scatter",s=50,label='test',ax=ax)
plt.legend()
plt.ylabel('ALBUMIN')
plt.xlabel('PROTIME')
plt.show()
ax=training_data.plot(x='ASCITES',y='ALBUMIN',c='Class', cmap="viridis",marker="*", kind="scatter",s=50,label='train')
testing_data.plot(x='ASCITES',y='ALBUMIN',c='Class', cmap="viridis",marker="o", kind="scatter",s=50,label='test',ax=ax)
plt.legend()
plt.ylabel('ALBUMIN')
plt.xlabel('ASCITES')
plt.show()
ax=training_data.plot(x='ASCITES',y='PROTIME',c='Class', cmap="viridis",marker="*", kind="scatter",s=50,label='train')
testing_data.plot(x='ASCITES',y='PROTIME',c='Class', cmap="viridis",marker="o", kind="scatter",s=50,label='test',ax=ax)
plt.legend()
plt.ylabel('PROTIME')
plt.xlabel('ASCITES')
plt.show()
ax=training_data.plot(x='ASCITES',y='HISTOLOGY',c='Class', cmap="viridis",marker="*", kind="scatter",s=50,label='train')
testing_data.plot(x='ASCITES',y='HISTOLOGY',c='Class', cmap="viridis",marker="o", kind="scatter",s=50,label='test',ax=ax)
plt.legend()
plt.ylabel('HISTOLOGY')
plt.xlabel('ASCITES')
plt.show()
ax=training_data.plot(x='PROTIME',y='HISTOLOGY',c='Class', cmap="viridis",marker="*", kind="scatter",s=50,label='train')
testing_data.plot(x='PROTIME',y='HISTOLOGY',c='Class', cmap="viridis",marker="o", kind="scatter",s=50,label='test',ax=ax)
plt.legend()
plt.ylabel('HISTOLOGY')
plt.xlabel('PROTIME')
plt.show()
ax=training_data.plot(x='ALBUMIN',y='HISTOLOGY',c='Class', cmap="viridis",marker="*", kind="scatter",s=50,label='train')
testing_data.plot(x='ALBUMIN',y='HISTOLOGY',c='Class', cmap="viridis",marker="o", kind="scatter",s=50,label='test',ax=ax)
plt.legend()
plt.ylabel('HISTOLOGY')
plt.xlabel('ALBUMIN')
plt.show()

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


plt.scatter(x_train[:,1], x_train[:,0], c=y_train, marker='o', label='train')
plt.scatter(x_test[:,1], x_test[:,0], c=y_test, marker='s', label='test')
plt.legend()
plt.ylabel('ALBUMIN')
plt.xlabel('PROTIME')
plt.show()

#Task2
#KNN ALGORITHM
#Euclidean Distance
euclidean = lambda x1, x2: np.sqrt(np.sum((x1 - x2)**2,axis = -1))
manhattan = lambda x1, x2: np.sum(np.abs(x1 - x2), axis=-1)
class KNN:

    def __init__(self, K=1, dist_fn= euclidean):
        self.dist_fn = dist_fn                                                    
        self.K = K
        return
    
    def fit(self, x_np, y_np):
        self.x_np = x_np
        self.y_np = y_np
        self.C = C
        return self
    
    def predict(self, x_test):
        num_test = x_test.shape[0]
        distances = self.dist_fn(self.x_np[None,:,:], x_test[:,None,:])
        knns = np.zeros((num_test, self.K), dtype=int)
        y_prob = np.zeros((num_test, 3))
        for i in range(num_test):
            knns[i,:] = np.argsort(distances[i])[:self.K]  
            y_prob[i,:] = np.bincount(self.y_np[knns[i,:]], minlength=(self.C)+1) 
        y_prob /= self.K                                                          
        return y_prob, knns
    
#Task3
#1-Testing Accuracy    
model = KNN(K=6)
y_prob, knns = model.fit(x_train, y_train).predict(x_test)
y_pred = np.argmax(y_prob,axis=-1)                                                #This returns the indeces of the largest element in the array
accuracy = np.sum(y_pred == y_test)/y_test.shape[0]
print(f'Testing accuracy is {accuracy*100:.1f}')


correct = y_test == y_pred
incorrect = np.logical_not(correct)


plt.scatter(x_train[:,0], x_train[:,1], c=y_train, marker='o', alpha=.2, label='train')
plt.scatter(x_test[correct,0], x_test[correct,1], marker='.', c=y_pred[correct], label='correct')
plt.scatter(x_test[incorrect,0], x_test[incorrect,1], marker='x', c=y_test[incorrect], label='misclassified')


#2-Decision boundry
for i in range(x_test.shape[0]):
    for k in range(model.K):
        hor = x_test[i,0], x_train[knns[i,k],0]
        ver = x_test[i,1], x_train[knns[i,k],1]
        plt.plot(hor, ver, 'k-', alpha=.1)
    

x0v = np.linspace(np.min(x_np[:,0]), np.max(x_np[:,0]), 200)
x1v = np.linspace(np.min(x_np[:,1]), np.max(x_np[:,1]), 200) 
x0, x1 = np.meshgrid(x0v, x1v)
x_all = np.vstack((x0.ravel(),x1.ravel())).T


for k in range(1,8):
  model = KNN(K=k)

  y_train_prob = np.zeros((y_train.shape[0], C+1))
  y_train_prob[np.arange(y_train.shape[0]), y_train] = 1

  
  y_prob_all, _ = model.fit(x_train, y_train).predict(x_all)

  y_pred_all = np.zeros_like(y_prob_all)
  y_pred_all[np.arange(x_all.shape[0]), np.argmax(y_prob_all, axis=-1)] = 1

  plt.scatter(x_train[:,1], x_train[:,0], c=y_train_prob, marker='o', alpha=1)
  plt.scatter(x_all[:,1], x_all[:,0], c=y_pred_all, marker='.', alpha=0.01)
  plt.ylabel('ALBUMIN')
  plt.xlabel('PROTIME')
  plt.show()   
  
  
#3-Training accuracy
model = KNN(K=6)
y_prob, knns = model.fit(x_train, y_train).predict(x_train)
y_pred = np.argmax(y_prob,axis=-1)                                                #This returns the indeces of the largest element in the array
accuracy = np.sum(y_pred == y_train)/y_train.shape[0]
print(f'Training accuracy is {accuracy*100:.1f}')

correct = y_train == y_pred
incorrect = np.logical_not(correct)

plt.scatter(x_train[:,0], x_train[:,1], c=y_train, marker='o', alpha=.2, label='train')
plt.scatter(x_train[correct,0], x_train[correct,1], marker='.', c=y_pred[correct], label='correct')
plt.scatter(x_train[incorrect,0], x_train[incorrect,1], marker='x', c=y_train[incorrect], label='misclassified')


 
#5-Validation accuracy
model = KNN(K=6)
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
y_prob, knns = model.fit(S12np, yX12npf).predict(S12np)
y_pred = np.argmax(y_prob,axis=-1)                                                #This returns the indeces of the largest element in the array
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
y_prob, knns = model.fit(S22np, yX22npf).predict(S22np)
y_pred = np.argmax(y_prob,axis=-1)                                                #This returns the indeces of the largest element in the array
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
y_prob, knns = model.fit(S42np, yX42npf).predict(S42np)
y_pred = np.argmax(y_prob,axis=-1)                                                #This returns the indeces of the largest element in the array
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
y_prob, knns = model.fit(S62np, yX62npf).predict(S62np)
y_pred = np.argmax(y_prob,axis=-1)                                                #This returns the indeces of the largest element in the array
accuracy4 = np.sum(y_pred == yS62npf)/yS62npf.shape[0]

average_Accuracy=((accuracy1+accuracy2+accuracy3+accuracy4)*100)/4
print('Validation accuracy=', average_Accuracy)

  















  
    
    