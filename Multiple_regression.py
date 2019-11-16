# -*- coding: utf-8 -*-
"""
Created on Tue Jan 22 23:59:27 2019

@author: kumar
"""

import pandas as pd 
import numpy as np 
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as pt
data=pd.read_csv("/home/curaj_bda/Desktop/ICECREAM.csv")
trn,tst=train_test_split(data,train_size=0.6)
trval=trn.values
tsval=tst.values
X=trval[0:5]
def GD(B,X,Y,alfa):
    m=len(Y)
    z=alfa/m
    B=B-(z*(((B.T.dot(X.T))-Y).dot(X)).T)
    return(B)
def Multiple_linear_regression(X,Y):
    B=np.zeroes(len(Y)+1)
    alfa=0.00001
    NOI=1000
    Thresh_error=0.001
    J=[]
    for i in range(NOI):
        yp=B.T.dot(X.T)
        er=yp-Y
        J.append((1/(2*len(Y))*np.sum(np.square(er))))
        if(J[i]<Thresh_error):
            break
        elif(len(J)>10 and np.mean(J[-10:])==J[-1]):
            break
        else:
            A=GD(B,X,Y,alfa)
    return(A,J)

    