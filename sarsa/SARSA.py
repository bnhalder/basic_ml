#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 21 15:06:28 2017

@author: jabong
"""

import numpy as np

def SARSA():
    R = np.array([[-5,0,-np.inf,-np.inf,-np.inf,-np.inf],[0,-5,0,0,-np.inf,-np.inf],[-np.inf,0,-5,0,-np.inf,100],[-np.inf,0,0,-5,0,-np.inf],[-np.inf,-np.inf,-np.inf,0,-5,100],[-np.inf,-np.inf,0,-np.inf,-np.inf,0]])
    t = np.array([[1,1,0,0,0,0],[1,1,1,1,0,0],[0,1,1,1,0,1],[0,1,1,1,1,0],[0,0,0,1,1,1],[0,0,1,0,1,1]])
    
    nStates = np.shape(R)[0]
    nActions = np.shape(R)[1]
    Q = np.random.rand(nStates,nActions)*0.1-0.05
    mu = 0.7
    gamma = 0.4
    epsilon = 0.1
    nits = 0
    
    while nits < 1000:
        s = np.random.randint(nStates)
        if (np.random.rand()<epsilon):
            indices = np.where(t[s,:]!=0)
            pick = np.random.randint(np.shape(indices)[1])
            a = indices[0][pick]
        else:
            a = np.argmax(Q[s,:])
        
        while s != 5:
            r = R[s, a]
            sprime = s
            if (np.random.rand()<epsilon):
                indices = np.where(t[sprime,:]!=0)
                pick = np.random.randint(np.shape(indices)[1])
                aprime = indices[0][pick]
                #print s,a
            else:
                aprime = np.argmax(Q[sprime,:])
            
            Q[s,a] += mu * (r + gamma*Q[sprime,aprime] - Q[s,a])

            s = sprime
            a = aprime
            
        nits += 1
    print Q
    
SARSA()