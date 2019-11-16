# -*- coding: utf-8 -*-
"""
Created on Wed Jan 23 22 23:59:27 2019

@author: Anurag sharma
"""

import pandas as pd 
import numpy as np 
import matplotlib.pyplot as pt
from sklearn.metrics import mean_squared_error
traindata=pd.read_csv("C:\\Users\\Anurag sharma\\Desktop\\CODE\\ML\\train.csv")
testdata=pd.read_csv("C:\\Users\\Anurag sharma\\Desktop\\CODE\\ML\\test.csv")

r=testdata.values
#print (r)
a=testdata.shape
#print (a)
test=np.matrix([np.ones(a[0]),r[:,0],r[:,1],r[:,2],r[:,3]]).T
#print (test)
trval=traindata.values.T
#print (trval)
Y=np.matrix(trval[4])
#print (Y)
X=np.matrix([np.ones(len(Y.T)),trval[0],trval[1],trval[2],trval[3]])
#print (X)
def grad_desc(B,X,Y,lr):
    m=len(Y)
    z=lr/m
    B=B-(z*(((B.T.dot(X))-Y).dot(X.T)).T)
    print(B)
    return(B)
def Multiple_linear_regression(X,Y):
    B=np.matrix([0.1,0.1,0.2,0.1,0.2])
    lr=0.0000000000001
    noi=100
    te=0.001
    cf=[]
    for i in range(noi):
        y_p=B.dot(X)
        error=y_p-Y
        cf.append((1/(2*len(Y))*np.sum(np.square(error))))
        if(cf[i]<te):
            break
        elif(len(cf)>10 and np.mean(cf[-10:])==cf[-1]):
            break
        else:
            B=grad_desc(B.T,X,Y,lr).T
    return(cf,B)
def testing(X,Y,B):
    y_p= B.dot(X)
    rmserror=np.sqrt(mean_squared_error(Y,y_p))
    return(rmserror)
def plot_error(cf):
    a=[]
    for i in range(len(cf)):
        a.append(i)
    pt.plot(a,cf)

def contour_plot(j,x,y):
    pt.contour(j,x,y,colour='black') 
    
    
