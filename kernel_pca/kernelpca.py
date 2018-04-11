#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 15 17:49:20 2017

@author: jabong
"""

import numpy as np
import pylab as pl

def kernelmatrix(data, kernel, param=np.array([3,2])):
    if kernel == 'linear':
        return np.dot(data, np.transpose(data))
    elif kernel == 'gaussian':
        K = np.zeros((data.shape[0], data.shape[0]))
        for i in range(data.shape[0]):
            for j in range(i+1, data.shape[0]):
                K[i,j] = np.sum((data[i,:] - data[j,:])**2)
                K[j,i] = K[i,j]
        return np.exp(-K**2/(2*param[0]**2))
    elif kernel == 'polynomial':
        return (np.dot(data, np.transpose(data))+param[0])**param[1]

def kernelpca(data, kernel, redDim):
    nData = data.shape[0]
    
    K = kernelmatrix(data, kernel)
    D = np.sum(K,axis=0)/nData
    E = np.sum(D)/nData
    J = np.ones((nData,1))*D
    K = K - J - np.transpose(J) + E*np.ones((nData,nData))
    
    evals,evecs = np.linalg.eig(K) 
    indices = np.argsort(evals)
    indices = indices[::-1]
    evecs = evecs[:,indices[:redDim]]
    evals = evals[indices[:redDim]]
    
    sqrtE = np.zeros((len(evals),len(evals)))
    for i in range(len(evals)):
        sqrtE[i,i] = np.sqrt(evals[i])
    
    newData = np.transpose(np.dot(sqrtE,np.transpose(evecs)))
    return newData

#data = np.array([[0.1,0.1],[0.2,0.2],[0.3,0.3],[0.35,0.3],[0.4,0.4],[0.6,0.4],[0.7,0.45],[0.75,0.4],[0.8,0.35]])
#newData = kernelpca(data,'gaussian',2)
#pl.plot(data[:,0],data[:,1],'o',newData[:,0],newData[:,0],'.')
#show()
    

    