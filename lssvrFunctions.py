# -*- coding: utf-8 -*-
"""
Created on Thu May 16 17:52:05 2019

@author: XPS 15
"""
import numpy as np

def getKernelBlock(x,y,lam,c):
    n = x.shape[0]
    m = getKernelMatrix(x,y,lam)+ np.eye(n)/c
    return m

def getKernelMatrix(x,y,lam):
    if x.ndim >1:
        x2 = sum((x**2).T).reshape(x.shape[0],1)
        y2 = sum((y**2).T).reshape(y.shape[0],1)
    else:
        x2 = x*x
        y2 = y*y               
    norm = y2[:,...] + x2.T - 2*np.dot(y,x.T) 
    return np.exp(-norm*lam)

def compute_lssvr(x_sample,x_test,lam,alpha,b):   
    matrix = getKernelMatrix(x_sample,x_test,lam)
    output = np.dot(matrix,alpha)+b
    return output

def getParams(x_sample,y_sample,lam,c):
    n = y_sample.shape[0]
    matrix = getKernelBlock(x_sample,x_sample,lam,10)
    temp_v = np.row_stack((np.ones(n), matrix))
    temp_h = np.row_stack(([0],np.ones((n,1))))
    A = np.column_stack((temp_h,temp_v ) )
    temp_b = np.row_stack(([0],y_sample))
    params =  np.dot( np.linalg.pinv(A),temp_b )
    alpha = params[1:]
    bias = params[0]
    return alpha,bias
