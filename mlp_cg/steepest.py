#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 19 12:43:29 2017

@author: jabong
"""

import numpy as np

def Jacobian(x):
    return np.array([x[0], 0.4*x[1], 1.2*x[2]])

def steepest(x0):
    i = 0;
    iMax = 10;
    x = x0
    Delta = 1
    alpha = 1
    
    while i<iMax and Delta > 10**(-5):
        p = -Jacobian(x)
        xOld = x
        x = x + alpha*p
        Delta = np.sum((x-xOld)**2)
        print x
        i += 1


x0 = np.array([-2, 2, -2])
steepest(x0)