#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 15 17:34:37 2017

@author: jabong
"""

import numpy as np

def pca(data, redDim=0, normalise=1):
    m = np.mean(data, axis=0)
    data -= m
    C = np.cov(np.transpose(data))
    
    evals, evecs = np.linalg.eig(C)
    indices = np.argsort(evals)
    indices = indices[::-1]
    evecs = evecs[:, indices]
    evals = evals[indices]
    
    if redDim > 0:
        evecs = evecs[:, :redDim]
    
    if normalise:
        for i in range(evecs.shape[1]):
            evecs[:,i] /= np.linalg.norm(evecs[:,i]) * np.sqrt(evals[i])
    
    # Produce the new data matrix
    x = np.dot(np.transpose(evecs),np.transpose(data))
    # Compute the original data again
    y = np.transpose(np.dot(evecs,x))+m
    return x, y, evals, evecs