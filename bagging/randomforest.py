#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 22 16:29:59 2017

@author: jabong
"""

import numpy as np
import dtree

class randomforest:
    
    def __init__(self):
        self.tree = dtree.dtree()
    
    def rf(self, data, targets, features, nTrees, nSamples, nFeatures, maxlevel=5):
        nPoints = np.shape(data)[0]
        nDim = np.shape(data)[1]
        self.nSamples = nSamples
        self.nTrees = nTrees
        
        classifiers = []
        
        for i in range(nTrees):
            print i
            samplePoints = np.random.randint(0, nPoints, (nPoints, nSamples))
            
            for j in range(nSamples):
                sample = []
                sampleTarget = []
                for k in range(nPoints):
                    sample.append(data[samplePoints[k,j]])
                    sampleTarget.append(targets[samplePoints[k,j]])
            classifiers.append(self.tree.make_tree(sample,sampleTarget,features,maxlevel,forest=nFeatures))
        return classifiers
    
    def rfclass(self, classifiers, data):
        decision = []
        for j in range(len(data)):
            outputs = []
            for i in range(self.nTrees):
                out = self.tree.classify(classifiers[i],data[j])
                if out is not None:
                    outputs.append(out)
                
            out = []
            for each in outputs:
                if out.count(each)==0:
                    out.append(each)
            frequency = np.zeros(len(out))
            
            index = 0
            if len(out)>0:
                for each in out:
                    frequency[index] = outputs.count(each)
                    index += 1
                decision.append(out[frequency.argmax()])
            else:
                decision.append(None)
        return decision