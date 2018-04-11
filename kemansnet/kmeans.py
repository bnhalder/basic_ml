#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 22 17:22:41 2017

@author: jabong
"""

#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 22 16:59:42 2017

@author: jabong
"""

import numpy as np

class kmeans:
    
    def __init__(self, k, data):
        self.nData = np.shape(data)[0]
        self.nDim = np.shape(data)[1]
        self.k = k
    
    def kmeanstrain(self, data, maxIterations=10):
        minima = data.min(axis=0)
        maxima = data.max(axis=0)
        
        self.centres = np.random.rand(self.k,self.nDim)*(maxima-minima)+minima
        oldCentres = np.random.rand(self.k,self.nDim)*(maxima-minima)+minima
        
        count = 0
        while np.sum(np.sum(oldCentres-self.centres))!= 0 and count<maxIterations:
            oldCentres = self.centres.copy()
            count += 1
            
            distances = np.ones((1,self.nData))*np.sum((data-self.centres[0,:])**2,axis=1)
            for j in range(self.k-1):
                distances = np.append(distances,np.ones((1,self.nData))*np.sum((data-self.centres[j+1,:])**2,axis=1),axis=0)
            
            cluster = distances.argmin(axis=0)
            cluster = np.transpose(cluster*np.ones((1,self.nData)))
            
            for j in range(self.k):
                thisCluster = np.where(cluster==j,1,0)
                if sum(thisCluster)>0:
                    self.centres[j,:] = np.sum(data*thisCluster,axis=0)/np.sum(thisCluster)
            
        return self.centres
    
    def kmeansfwd(self, data):
        nData = np.shape(data)[0]
        distances = np.ones((1,nData))*np.sum((data-self.centres[0,:])**2,axis=1)
        for j in range(self.k-1):
            distances = np.append(distances,np.ones((1,nData))*np.sum((data-self.centres[j+1,:])**2,axis=1),axis=0)
        
        cluster = distances.argmin(axis=0)
        cluster = np.transpose(cluster*np.ones((1,nData)))
        
        return cluster
        
        
        