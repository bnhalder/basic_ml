#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 15 21:21:01 2017

@author: jabong
"""

import numpy as np

def knn(k, data, dataClass, inputs):
    nInputs = inputs.shape[0]
    closest = np.zeros(nInputs)
    
    for n in range(nInputs):
        distances = np.sum((data - inputs[n, :])**2, axis=1)
        indices = np.argsort(distances, axis=0)
        classes = np.unique(dataClass[indices[:k]])
        
        if len(classes) == 1:
            closest[n] = np.unique(classes)
        else:
            counts = np.zeros(max(classes) + 1)
            for i in range(k):
                counts[dataClass[indices[i]]] += 1
            closest[n] = np.max(counts)
    
    return closest
                