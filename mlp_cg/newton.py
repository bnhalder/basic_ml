#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 19 12:48:33 2017

@author: jabong
"""

import numpy as np

def Jacobian(x):
    return np.array([x[0], 0.4*x[1], 1.2*x[2]])

def Hessian(x):
    return np.array([[1, 0, 0], [0, 0.4, 0], [0, 0, 1.2]])

def Newton(x0):
    i = 0
    iMax = 10
    x = x0
    Delta = 1
    alpha = 1
    
    while i < iMax and Delta > 10**(-5):
        p = -np.dot(np.linalg.pinv(Hessian(x)), Jacobian(x))
        xOld = x
        x = x + alpha*p
        Delta = np.sum((x-xOld)**2)
        i += 1
    print x

x0 = np.array([-2, 2, -2])
Newton(x0)
    