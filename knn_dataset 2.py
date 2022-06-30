# -*- coding: utf-8 -*-
"""
Created on Sun Feb  6 02:50:24 2022

@author: taha
"""

import pandas as pd 

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_classif
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from IPython.core.debugger import set_trace 
from statistics import mode
import seaborn as sns
from sklearn.feature_selection import f_classif
from matplotlib import pyplot

#Task1
#Aquire, preprocess and analyze the data
#1-describe the data set related to DR data with 19 attributes(x)and 1151 instances
dataset = pd.read_csv(r'C:\Users\taha\Desktop\R.csv', names = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16", "17", "18", "Class"])
dataset.head()
print(dataset)
help(dataset)


#2-using simple statistics
#a-converting columns to string and numerical
dataset["0"] = pd.to_numeric(dataset["0"]) 
dataset["1"] = pd.to_numeric(dataset["1"])
dataset["2"] = pd.to_numeric(dataset["2"]) 
dataset["3"] = pd.to_numeric(dataset["3"])
dataset["4"] = pd.to_numeric(dataset["4"])
dataset["5"] = pd.to_numeric(dataset["5"])
dataset["6"]= pd.to_numeric(dataset["6"])
dataset["7"]= pd.to_numeric(dataset["7"])
dataset["8"]= pd.to_numeric(dataset["8"])
dataset["9"]= pd.to_numeric(dataset["9"])
dataset["10"]= pd.to_numeric(dataset["10"])
dataset["11"]= pd.to_numeric(dataset["11"])
dataset["12"]= pd.to_numeric(dataset["12"])
dataset["13"]= pd.to_numeric(dataset["13"])
dataset["14"]= pd.to_numeric(dataset["14"])
dataset["15"]= pd.to_numeric(dataset["15"])
dataset["16"]= pd.to_numeric(dataset["16"])
dataset["17"]= pd.to_numeric(dataset["17"])
dataset["18"]= pd.to_numeric(dataset["18"])
dataset["Class"]= pd.to_numeric(dataset["Class"])


#b-Specify features and class, find the distributions of the features and classes
x,y=dataset[["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16", "17", "18"]],dataset["Class"]   


#c-Specify training data and testing data                                         
training_data = dataset.sample(frac=0.8, random_state=25)
testing_data = dataset.drop(training_data.index)


#d-separating features and class in train and test data
x_train_1,y_train_1=training_data[["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16", "17", "18"]],training_data[["Class"]]
x_test_1,y_test_1=testing_data[["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16", "17", "18"]],testing_data[["Class"]]


#e-Using Pearson Correlation
plt.figure(figsize=(12,10))
cor = dataset.corr()
sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
plt.show()


#f-feature selection
def select_features(x_train_1, y_train_1, X_test_1):
	fs = SelectKBest(score_func=f_classif, k='all')
	fs.fit(x_train_1, y_train_1)
	X_train_fs = fs.transform(x_train_1)
	X_test_fs = fs.transform(x_test_1)
	return X_train_fs, X_test_fs, fs
X_train_fs, X_test_fs, fs = select_features(x_train_1, y_train_1, x_test_1)
for i in range(len(fs.scores_)):
	print('Feature %d: %f' % (i, fs.scores_[i]))
pyplot.bar([i for i in range(len(fs.scores_))], fs.scores_)
pyplot.show()

#g-visualization of the data
ax=training_data.plot(x='2',y='3',c='Class', cmap="viridis",marker="s", kind="scatter",s=50,label='train')
testing_data.plot(x='2',y='3',c='Class', cmap="viridis",marker="o", kind="scatter",s=50,label='test',ax=ax)
plt.legend()
plt.ylabel('3')
plt.xlabel('2')
plt.show()
ax=training_data.plot(x='2',y='4',c='Class', cmap="viridis",marker="s", kind="scatter",s=50,label='train')
testing_data.plot(x='2',y='4',c='Class', cmap="viridis",marker="o", kind="scatter",s=50,label='test',ax=ax)
plt.legend()
plt.ylabel('4')
plt.xlabel('2')
plt.show()
ax=training_data.plot(x='2',y='5',c='Class', cmap="viridis",marker="s", kind="scatter",s=50,label='train')
testing_data.plot(x='2',y='5',c='Class', cmap="viridis",marker="o", kind="scatter",s=50,label='test',ax=ax)
plt.legend()
plt.ylabel('5')
plt.xlabel('2')
plt.show()
ax=training_data.plot(x='3',y='4',c='Class', cmap="viridis",marker="s", kind="scatter",s=50,label='train')
testing_data.plot(x='3',y='4',c='Class', cmap="viridis",marker="o", kind="scatter",s=50,label='test',ax=ax)
plt.legend()
plt.ylabel('4')
plt.xlabel('3')
plt.show()
ax=training_data.plot(x='3',y='5',c='Class', cmap="viridis",marker="s", kind="scatter",s=50,label='train')
testing_data.plot(x='3',y='5',c='Class', cmap="viridis",marker="o", kind="scatter",s=50,label='test',ax=ax)
plt.legend()
plt.ylabel('5')
plt.xlabel('3')
plt.show()
ax=training_data.plot(x='4',y='5',c='Class', cmap="viridis",marker="s", kind="scatter",s=50,label='train')
testing_data.plot(x='4',y='5',c='Class', cmap="viridis",marker="o", kind="scatter",s=50,label='test',ax=ax)
plt.legend()
plt.ylabel('5')
plt.xlabel('4')
plt.show()

x_train_2f = x_train_1[["2","3","4","5"]]
x_test_2f = x_test_1[["2","3","4","5"]]
x_2f=x[["2","3","4","5"]]
x_final=x[["2","3"]]
x_final_np=x_final.to_numpy()
x_train_final=x_train_1[["2","3"]]
x_test_final=x_test_1[["2","3"]]
x_train_final_np=x_train_final.to_numpy()
x_test_final_np=x_test_1.to_numpy()
(N,D), C = x_2f.shape,int( np.max(y))


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
#KNN ALGORITHM
#Euclidean Distance
euclidean = lambda x1, x2: np.sqrt(np.sum((x1 - x2)**2,axis = -1))
manhattan = lambda x1, x2: np.sum(np.abs(x1 - x2), axis=-1)
class KNN:

    def __init__(self, K=1, dist_fn= manhattan):
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
            y_prob[i,:] = np.bincount(self.y_np[knns[i,:]], minlength=(self.C)+2) 
        y_prob /= self.K                                                          
        return y_prob, knns
  
#Task3
#1-Testing Accuracy
model = KNN(K=3)
y_prob, knns = model.fit(x_train, y_train).predict(x_test)
y_pred = np.argmax(y_prob,axis=-1)                                                #This returns the indeces of the largest element in the array
accuracy = np.sum(y_pred == y_test)/y_test.shape[0]
print(f'Testing accuracy is {accuracy*100:.1f}.')

correct = y_test == y_pred
incorrect = np.logical_not(correct)

plt.scatter(x_train[:,0], x_train[:,1], c=y_train, marker='o', alpha=.2, label='train')
plt.scatter(x_test[correct,0], x_test[correct,1], marker='.', c=y_pred[correct], label='correct')
plt.scatter(x_test[incorrect,0], x_test[incorrect,1], marker='x', c=y_test[incorrect], label='misclassified')



#2-Training Accuracy
model = KNN(K=3)
y_prob, knns = model.fit(x_train, y_train).predict(x_train)
y_pred = np.argmax(y_prob,axis=-1)                                                #This returns the indeces of the largest element in the array
accuracy = np.sum(y_pred == y_train)/y_train.shape[0]
print(f'Training accuracy is {accuracy*100:.1f}.')

correct = y_train == y_pred
incorrect = np.logical_not(correct)

plt.scatter(x_train[:,0], x_train[:,1], c=y_train, marker='o', alpha=.2, label='train')
plt.scatter(x_train[correct,0], x_train[correct,1], marker='.', c=y_pred[correct], label='correct')
plt.scatter(x_train[incorrect,0], x_train[incorrect,1], marker='x', c=y_train[incorrect], label='misclassified')



#3-Validation Accuracy
model = KNN(K=3)
S1 = training_data.sample(frac=0.25, random_state=25)
X1=training_data.drop(S1.index)
S12,yS12=S1[[ "2", "3", "4", "5"]],S1[["Class"]]
X12,yX12=X1[["2", "3", "4", "5"]],X1[["Class"]]
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
S22,yS22=S2[["2", "3", "4", "5"]],S2[["Class"]]
X22,yX22=X2[["2", "3", "4", "5"]],X2[["Class"]]
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
S42,yS42=S4[["2", "3", "4", "5"]],S4[["Class"]]
X42,yX42=X4[["2", "3", "4", "5"]],X4[["Class"]]
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
S62,yS62=S6[["2", "3", "4", "5"]],S6[["Class"]]
X62,yX62=X6[["2", "3", "4", "5"]],X6[["Class"]]
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

average_accuracy=((accuracy1+accuracy2+accuracy3+accuracy4)*100)/4
print('Validation Accuracy=', average_accuracy)


#4-Decision boundry
for i in range(x_test_final_np.shape[0]):
    for k in range(model.K):
        hor = x_test_final_np[i,0], x_train_final_np[knns[i,k],0]
        ver = x_test_final_np[i,1], x_train_final_np[knns[i,k],1]
        plt.plot(hor, ver, 'k-', alpha=.1)
    

x0v = np.linspace(np.min(x_final_np[:,0]), np.max(x_final_np[:,0]), 200)
x1v = np.linspace(np.min(x_final_np[:,1]), np.max(x_final_np[:,1]), 200) 
x0, x1 = np.meshgrid(x0v, x1v)
x_all = np.vstack((x0.ravel(),x1.ravel())).T


for k in range(1,4):
  model = KNN(K=k)

  y_train_prob = np.zeros((y_train.shape[0], C+2))
  y_train_prob[np.arange(y_train.shape[0]), y_train] = 1

  
  y_prob_all, _ = model.fit(x_train_final_np, y_train).predict(x_all)

  y_pred_all = np.zeros_like(y_prob_all)
  y_pred_all[np.arange(x_all.shape[0]), np.argmax(y_prob_all, axis=-1)] = 1
 
  plt.scatter(x_train_final_np[:,1], x_train_final_np[:,0],c=y_train_prob, marker='o', alpha=1)
  plt.scatter(x_all[:,1], x_all[:,0], c=y_pred_all, marker='.', alpha=0.01)
  plt.ylabel('2')
  plt.xlabel('3')
  plt.show()    

  
  
  
  
  
  
  
  
  
  
  
  
  
  