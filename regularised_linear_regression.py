# -*- coding: utf-8 -*-
"""
Created on Mon Jan 26 01:09:57 2019

@author: Anurag sharma
"""

import pandas as pd 
import numpy as np 
import matplotlib.pyplot as pt
from sklearn.metrics import mean_squared_error
import sklearn.preprocessing as sklearn
xyz=pd.read_csv("C:\\Users\\Anurag sharma\\Desktop\\ML\\train.csv")
sc=sklearn.StandardScaler()
data=(xyz.values).T
tdata=pd.read_csv("C:\\Users\\Anurag sharma\\Desktop\\ML\\testnew.csv")
r=(tdata.values).T
yt=r[4]
b=r.shape
a=data.shape
XT=np.array([np.ones(b[1]),r[0],r[1],r[2],r[3]])
Y=np.array(data[4])
X=np.array([np.ones(a[1]),data[0],data[1],data[2],data[3]])
def grad_desc(B,X,Y,alfa,lamda):
    m=len(Y)
    z=alfa/m
    B=B-(z*((((B.T.dot(X))-Y).dot(X.T)).T)+lamda*B)
    return(B)
def Regularized_regression(X,Y):
    B=np.matrix([0.1,0.1,0.2,0.1,0.2])
    lr=0.00000001
    lamda=0.00001
    noi=1000
    te=0.001
    cf=[]
    for i in range(noi):
        y_p=B.dot(X)
        error=y_p-Y
        cf.append((1/(2*len(Y))*((np.sum(np.square(error)))+lamda*np.sum(np.square(B)))))
        if(cf[i]<te):
            break
        elif(len(cf)>10 and np.mean(cf[-10:])==cf[-1]):
            break
        else:
            B=grad_desc(B.T,X,Y,lr,lamda).T
    return(cf,B)
    
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


def contour_plot(j,x,y):
    pt.contour(j,x,y,colour='black')  
m=Regularized_regression(X,Y)
print('values of theta are:',m[1])
i=predict_y(XT.T,m[1].T) 
print('Predicted Y values :',i)
print('Cost vs No.of Iterations:')
plot_error(m[0])   

t = testing(XT.T,yt,m[1].T)
print ('error is:', t )