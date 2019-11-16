# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
'''this is the complete lab work on simple linear regression'''

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split as tts
from array import *

train = pd.read_csv("C:\\Users\\Anurag sharma\\Desktop\\CODE\\ML\\LabExam\\train.csv")
test=pd.read_csv("C:\\Users\\Anurag sharma\\Desktop\\CODE\\ML\\LabExam\\Test.csv")
x = train.drop(["Avg_Number_Icecream_sold_per_day","Customer_ID"],axis=1)
y = train.iloc[:,2:]
testdata1 = test.drop(["Customer_ID"],axis=1)

m=len(x)
itre=[] 
mval=[]
'''def plotting(x,y):
    return(plt.plot(x,y))
def scatter():
    return(plt.plot(x,y,'r+'))'''
def lin_reg(x,y):
    lr= 0.0001
    te= 0.003
    noi=1000
    theta0 = 1
    theta1 = 1
    cf_list = []
    m=len(x)
    for i in range(noi):
        y_p = theta0*np.exp(x)+theta1
        error = y - y_p
        cf = ((1/(2*m))*(np.sum(np.square(error))))
        cf_list.append(cf)
        itre.append(i)
        if(cf_list[i]<te):
            break
        elif (len(cf_list)>10 and np.mean(cf_list[-10:])==cf_list[-1]):
            break
        else:
            A= grad_desc(x,y,lr,m,theta0,theta1)
            theta0,theta1=A[0],A[1]
    print(plotting(itre,cf_list))
    print(plotting)
    print(plt.plot())
    return(theta0,theta1,cf_list[-1])
        
def grad_desc(x,y,lr,m,theta0,theta1):
    for j in range (m):
        mval.append(j)
        j1=((1/m)*(theta0*x[j]+theta1-y[j])*np.exp(x[j]))
        j2=((1/m)*(theta0*x[j]+theta1-y[j]))
        theta0 = theta0 - (lr*j1)
        theta1 = theta1 - (lr*j2)
    return theta0,theta1
def testing(testval,theta0,theta1):
    x,y=testval[:,0],testval[:,1]
    one=np.ones(len(x))
    y_p=theta0*x+theta1*one
    err=y-y_p
    mr=np.sqrt(np.sum(np.square(err)))
    return(mr)
def plot_contour(cf_list,xy):
    plt.contour(cf_list,xy,colour='black')   
    
    

    
    
  
    
    
    
    
    