#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 11 19:40:48 2017

@author: jabong
"""

import numpy as np
import pcn
import kmeans

class rbf:
    
    def __init__(self, inputs, targets, nRBF, sigma=0, usekmeans=0, normalise=0):
        self.nin = inputs.shape[1]
        self.nout = targets.shape[1]
        self.ndata = inputs.shape[0]
        self.nRBF = nRBF
        self.usekmeans = usekmeans
        self.normalise = normalise
        
        if usekmeans:
            self.kmeansnet = kmeans.kmeans(self.nRBF, inputs)
        
        self.hidden = np.zeros((self.ndata, self.nRBF+1))
        
        if sigma==0:
            d = (inputs.max(axis=0) - inputs.min(axis=0)).max()
            self.sigma = d/np.sqrt(2*nRBF)
        else:
            self.sigma = sigma
            
        self.perceptron = pcn.pcn(self.hidden[:, :-1], targets)
        self.weights1 = np.zeros((self.nin, self.nRBF))
    
    def rbftrain(self, inputs, targets, eta=0.25, niterations=100):
        
        if self.usekmeans == 0:
            indices = range(self.ndata)
            np.random.shuffle(indices)
            for i in range(self.nRBF):
                self.weights1[:, i] = inputs[indices[i], :]
        else:
            self.weights1 = np.transpose(self.kmeansnet.kmeanstrain(inputs))
        
        for i in range(self.nRBF):
            self.hidden[:, i] = np.exp(-np.sum((inputs - np.ones((1, self.nin))*self.weights1[:, i])**2, axis=1)/(2*self.sigma**2))

        if self.normalise:
            self.hidden[:, :-1] /= np.transpose(np.ones((1,np.shape(self.hidden)[0]))*self.hidden[:,:-1].sum(axis=1))
        
        self.perceptron.pcntrain(self.hidden[:,:-1],targets,eta,niterations)
        
    def rbffwd(self, inputs):
        hidden = np.zeros((np.shape(inputs)[0],self.nRBF+1))
        for i in range(self.nRBF):
            hidden[:,i] = np.exp(-np.sum((inputs - np.ones((1,self.nin))*self.weights1[:,i])**2,axis=1)/(2*self.sigma**2))
        
        if self.normalise:
            hidden[:,:-1] /= np.transpose(np.ones((1,np.shape(hidden)[0]))*hidden[:,:-1].sum(axis=1))
        
        hidden[:,-1] = -1

        outputs = self.perceptron.pcnfwd(hidden)
        return outputs
    
    def confmat(self,inputs,targets):
        """Confusion matrix"""

        outputs = self.rbffwd(inputs)
        nClasses = np.shape(targets)[1]

        if nClasses==1:
            nClasses = 2
            outputs = np.where(outputs>0,1,0)
        else:
            # 1-of-N encoding
            outputs = np.argmax(outputs,1)
            targets = np.argmax(targets,1)

        cm = np.zeros((nClasses,nClasses))
        for i in range(nClasses):
            for j in range(nClasses):
                cm[i,j] = np.sum(np.where(outputs==i,1,0)*np.where(targets==j,1,0))

        print cm
        print np.trace(cm)/np.sum(cm)
                
        