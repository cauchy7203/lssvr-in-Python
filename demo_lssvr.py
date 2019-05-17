# -*- coding: utf-8 -*-
"""
Created on Wed May 15 14:22:21 2019

@author: XPS 15
"""
import numpy as np
import matplotlib.pyplot as plt 
import datetime

# here you need to add the path where you put the file 'lssvmFunctions.py'
import sys
sys.path.append('F:\PythonFiles')


# preparing the training data

n = 100
dim = 1


x = np.array(np.linspace(1,10,n)).reshape(n,dim)
y = np.sin(x)

# set the hyperparameters

sigma = .1
c = 100


#start to record runtime

starttime = datetime.datetime.now()


# get the parameters

alpha,bias = getParams(x,y,1/sigma**2,c)

# compute the output for training

y_pred =  compute_lssvr(x,x,1/sigma**2,alpha,bias)


# generate some points for testing

xtest = np.array(np.linspace(2,8,n)).reshape(n,dim)
ytest = compute_lssvr(x,xtest,1/sigma**2,alpha,bias)


# print the runtime

endtime = datetime.datetime.now()
print( 'Total runtime is:' )
print( (endtime - starttime).seconds)

# plot the results
plt.scatter(x,y,label='Raw data')
plt.scatter(x,y_pred,label='estimation')
plt.scatter(xtest,ytest,label='test')
plt.legend()
plt.show()

