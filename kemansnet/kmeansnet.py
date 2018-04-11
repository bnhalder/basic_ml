#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 22 17:24:22 2017

@author: jabong
"""

import numpy as np

class kmeans:
    
    def __init__(self, k, data, nEpochs=1000, eta=025):
        self.nData = np.shape(data)[0]
        self.nDim = np.shape(data)[1]
        self.k = k
        self.nEpochs = nEpochs
        self.weights = np.random.rand(self.nDim, self.k)
        self.eta = eta
    
    def kmeansfwd(self, data):
        best = np.zeros(np.shape(data)[0])
        for i in range(np.shape(data)[0]):
            activation = np.sum(self.weights*np.transpose(data[i:i+1,:]), axis=0)
            best[i] = np.argmax(activation)
        return best
    
    def kmeanstrain(self, data):
        normalisers = np.sqrt(np.sum(data**2,axis=1))*np.ones((1,np.shape(data)[0]))
        data = np.transpose(np.transpose(data)/normalisers)
        
        for i in range(self.nEpochs):
            for j in range(self.nData):
                activation = np.sum(self.weights*np.transpose(data[j:j+1,:]),axis=0)
                winner = np.argmax(activation)
                self.weights[:, winner] += self.eta * data[j, :] - self.weights[:, winner]
                
        