#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 21 14:52:31 2017

@author: jabong
"""

import numpy as np

def TDZero():
    R = np.array([[-5,0,-np.inf,-np.inf,-np.inf,-np.inf],[0,-5,0,0,-np.inf,-np.inf],[-np.inf,0,-5,0,-np.inf,100],[-np.inf,0,0,-5,0,-np.inf],[-np.inf,-np.inf,-np.inf,0,-5,100],[-np.inf,-np.inf,0,-np.inf,-np.inf,0]])
    t = np.array([[1,1,0,0,0,0],[1,1,1,1,0,0],[0,1,1,1,0,1],[0,1,1,1,1,0],[0,0,0,1,1,1],[0,0,1,0,1,1]])
    
    nStates = np.shape(R)[0]
    nActions = np.shape(R)[1]
    Q = np.random.rand(nStates, nActions)*0.1-0.05
    mu = 0.7
    gamma = 0.4
    epsilon = 0.1
    nits = 0
    
    while nits < 10000:
        s = np.random.randint(nStates)
        while s != 5:
            if(np.random.rand()<epsilon):
                indices = np.where(t[s, :]!=0)
                pick = np.random.randint(np.shape(indices)[1])
                a = indices[0][pick]
            else:
                a = np.argmax(Q[s,:])
            
            r = R[s, a]
            sprime = a
            Q[s,a] += mu * (r + gamma*np.max(Q[sprime,:]) - Q[s, a])
            s = sprime
        nits += 1
    print Q


TDZero()