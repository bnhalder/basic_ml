#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  6 17:52:24 2017

@author: jabong
"""

import numpy as np

class pcn:
    
    def __init__(self, inputs, targets):
        if np.ndim(inputs) > 1:
            self.nIn = np.shape(inputs)[1]
        else:
            self.nIn = 1
        
        if np.ndim(targets) > 1:
            self.nOut = targets.shape[1]
        else:
            self.nOut = 1
        
        self.nData = inputs.shape[0]
        self.weights = np.random.rand(self.nIn+1, self.nOut)*0.1-0.05
    
    def pcnfwd(self, inputs):
        activations = np.dot(inputs, self.weights)
        return np.where(activations > 0, 1, 0)
    
    def pcntrain(self, inputs, targets, eta, nIteration):
        inputs = np.concatenate((inputs, -np.ones((self.nData, 1))), axis=1)
        
        for n in range(nIteration):
            self.activations = self.pcnfwd(inputs)
            self.weights -= eta * np.dot(np.transpose(inputs), self.activations - targets)

    def confmat(self, inputs, targets):
        inputs = np.concatenate((inputs, -np.ones((self.nData, 1))), axis=1)
        outputs = np.dot(inputs, self.weights)
        nClasses = np.shape(targets)[1]
        
        if nClasses == 1:
            nClasses = 2
            outputs = np.where(outputs>0,1,0)
        else:
            outputs = np.argmax(outputs, 1)
            targets = np.argmax(targets, 1)
            
        cm = np.zeros((nClasses, nClasses))
        for i in range(nClasses):
            for j in range(nClasses):
                cm[i,j] = np.sum(np.where(outputs==i,1,0)*np.where(targets==j,1,0))
        
        print cm
        print np.trace(cm)/np.sum(cm)

def logic():
	import pcn
	""" Run AND and XOR logic functions"""

	a = np.array([[0,0,0],[0,1,0],[1,0,0],[1,1,1]])
	b = np.array([[0,0,0],[0,1,1],[1,0,1],[1,1,0]])

	p = pcn.pcn(a[:,0:2],a[:,2:])
	p.pcntrain(a[:,0:2],a[:,2:],0.25,10)
	p.confmat(a[:,0:2],a[:,2:])

	q = pcn.pcn(b[:,0:2],b[:,2:])
	q.pcntrain(b[:,0:2],b[:,2:],0.25,10)
	q.confmat(b[:,0:2],b[:,2:])
