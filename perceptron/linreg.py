#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  7 14:59:37 2017

@author: jabong
"""

import numpy as np

def linreg(inputs, targets):
    inputs = np.concatenate((inputs, np.ones((inputs.shape[0], 1))), axis = 1)
    beta = np.dot(np.dot(np.linalg.pinv(np.dot(np.transpose(inputs), inputs)), np.transpose(inputs)), targets)
    #outputs = np.dot(inputs, beta)
    return beta


def test():
    inputs = np.array([[0,0],[0,1],[1,0],[1,1]])
    testin = np.concatenate((inputs, np.ones((inputs.shape[0], 1))), axis = 1)
    # AND data
    ANDtargets = np.array([[0],[0],[0],[1]])
    # OR data
    ORtargets = np.array([[0],[1],[1],[1]])
    # XOR data
    XORtargets = np.array([[0],[1],[1],[0]])
    
    print "AND data"
    ANDbeta = linreg(inputs,ANDtargets)
    ANDout = np.dot(testin,ANDbeta)
    print ANDout
    
    print "OR data"
    ORbeta = linreg(inputs,ORtargets)
    ORout = np.dot(testin,ORbeta)
    print ORout
    
    print "XOR data"
    XORbeta = linreg(inputs,XORtargets)
    XORout = np.dot(testin,XORbeta)
    print XORout

    