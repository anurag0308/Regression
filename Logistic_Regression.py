# -*- coding: utf-8 -*-
"""
Created on Mon Feb 18 22:50:39 2019

@author: Anurag sharma
"""

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler

traindata1 = pd.read_csv("C:\\Users\\Anurag sharma\\Desktop\\CODE\\ML\\LogisticRegression\\train.csv")
testdata1 = pd.read_csv("C:\\Users\\Anurag sharma\\Desktop\\CODE\\ML\\LogisticRegression\\test.csv")
traindata = traindata1.drop(["Name","Id"],axis=1)
testdata = testdata1.drop(["Name","Id"],axis=1)

#from sklearn.preprocessing import MinMaxScaler
mean =traindata['3P%'].mean()
traindata['3P%'].fillna(mean,inplace=True)

mean =testdata['3P%'].mean()
testdata['3P%'].fillna(mean,inplace=True)

#from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
t = scaler.fit_transform(traindata)
a = t.shape
X=np.matrix([np.ones(a[0]),t[:,0],t[:,1],t[:,2],t[:,3],t[:,4],t[:,5],t[:,6],t[:,7],t[:,8],t[:,9],t[:,10],t[:,11],t[:,12],t[:,13],t[:,14],t[:,15],t[:,16],t[:,17],t[:,18]])

m = X.shape[1]

#scaler = MinMaxScaler()
t1 = scaler.fit_transform(testdata)
#Y = testval
Y=np.matrix([np.ones(t1.shape[0]),t1[:,0],t1[:,1],t1[:,2],t1[:,3],t1[:,4],t1[:,5],t1[:,6],t1[:,7],t1[:,8],t1[:,9],t1[:,10],t1[:,11],t1[:,12],t1[:,13],t1[:,14],t1[:,15],t1[:,16],t1[:,17],t1[:,18]])


def sigmoid(z):
    return 1/(1+np.exp(-z))

#def cost(h,y):
   # return -(y.dot(np.log(h))+(1-y).dot(np.log(1-h)))/m

def logistic_regression(X):
    lr=0.001
    noi=1000
    y= np.matrix(t[:,19])
    #m = X.shape[0]
    theta = np.zeros([1,20])
    te=0.001   
    cf_list=[]
    for i in range(noi):
        
        z= np.dot(X.T,theta.T)
        h = sigmoid(z) #hypothesis
        #cf = cost (h,y)
        cf=-(np.sum((y.dot(np.log(h))+(1-y).dot(np.log(1-h)))))/m
        cf_list.append(cf)
        if(cf_list[i]<te):
            break
        elif(len(cf_list)>10 and np.mean(cf_list[-10:])==cf_list[-1]):
            break
        else:
            theta=grad_desc(theta,lr,X)
    return(cf_list,theta)
    
def grad_desc(theta,lr,X):
    y= t[:,19]
    #m = X.shape[1]
    z= sigmoid(np.dot(X.T,theta.T))
    c=(z.T-y).dot(X.T)
    #for j in range (y):
    theta = theta - (lr/m)*c  
    return theta
def predict(theta,Y):
    prob=sigmoid(theta@Y)
    values=np.where(prob>=0.5,1,0)
    return np.squeeze(values)
cst,theta=logistic_regression(X)
id=testdata1['Id']
df=pd.DataFrame(predict(theta,Y),id)
df.to_csv('C:\\Users\\Anurag sharma\\Desktop\\CODE\\ML\\submission.csv')