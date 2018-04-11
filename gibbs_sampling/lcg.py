#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 16:06:06 2017

@author: jabong
"""

import numpy as np

def lcg(x0, n):
    a = 23
    m = 197
    c = 0
    
    rnd = np.zeros((n))
    rnd[0] = x0
    for i in range(1,n):
        rnd[i] = np.mod(a*rnd[i-1]+c, m)
        
    return rnd

print lcg(3,80) 