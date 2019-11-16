# -*- coding: utf-8 -*-
"""
Created on Tue Feb 12 00:00:47 2019

@author: Anurag sharma
"""

import pandas as pd 
import numpy as np 
import matplotlib.pyplot as pt
from sklearn.metrics import mean_squared_error
import sklearn.preprocessing as sklearn
data1=pd.read_csv("C:\\Users\\Anurag sharma\\Desktop\\ML\\train.csv")
traindata = (data1.values).T
data2=pd.read_csv("C:\\Users\\Anurag sharma\\Desktop\\ML\\testnew.csv")
testdata = (data2.values).T

b=testdata.shape
a=traindata.shape
actualY=testdata[4]

XT=np.array([np.ones(b[1]),testdata[0],testdata[1],testdata[2],testdata[3]])
Y=np.array(traindata[4])
X=np.array([np.ones(a[1]),traindata[0],traindata[1],traindata[2],traindata[3]])

def Lasso_regression(X,Y):
    B=np.matrix([0.1,0.1,0.2,0.1,0.2])
    lr=0.00000001
    lmbda=0.0001
    noi=1000
    te=0.001
    cf=[]
    for i in range(noi):
        y_p=B.dot(X)
        error=y_p-Y
        cf.append((1/(2*len(Y))*np.sum(np.square(error)+lmbda*np.sum(B))))
        if(cf[i]<te):
            break
        elif(len(cf)>10 and np.mean(cf[-10:])==cf[-1]):
            break
        else:
            B=grad_desc(B.T,X,Y,lr).T
    return(cf,B)
    
    
def grad_desc(B,X,Y,lr):
    lmbda= 0.0001
    m=len(Y)
    z=lr/m
    B=B-(z*(((B.T.dot(X))-Y).dot(X.T)).T+lmbda)
    print(B)
    return(B)
    
def plot_error(cf):
    a=[]
    for i in range(len(cf)):
        a.append(i)
    pt.plot(a,cf)
    
def predict_y(X,B):
    y=X@B
    return(y)    

def testing(X,Y,B):
    #y_p=B.dot(X)
    y_p=X@B
    rmserror=np.sqrt(mean_squared_error(Y,y_p))
    return(rmserror)
    
m=Lasso_regression(X,Y)
print('values of theta are:',m[1])

print('Cost vs No.of Iterations:')
plot_error(m[0]) 

i=predict_y(XT.T,m[1].T) 
print('Predicted Y values :',i)

t = testing(XT.T,actualY,m[1].T)
print ('error is:', t )


