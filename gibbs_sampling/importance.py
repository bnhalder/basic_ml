#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 16:16:40 2017

@author: jabong
"""

import pylab as pl
import numpy as np

def qsample():
    return np.random.rand()*4

def p(x):
    return 0.3*np.exp(-(x-0.3)**2) + 0.7*np.exp(-(x-2.)**2/0.3)

def q(x):
    return 4.0

def importance(nsamples):
    samples = np.zeros(nsamples, dtype=float)
    w = np.zeros(nsamples, dtype=float)
    
    for i in range(nsamples):
        samples[i] = qsample()
        w[i] = p(samples[i])/q(samples[i])
        
    return samples, w

x = np.arange(0,4,0.01)
x2 = np.arange(-0.5,4.5,0.1)
realdata = 0.3*np.exp(-(x-0.3)**2) + 0.7* np.exp(-(x-2.)**2/0.3) 
box = np.ones(len(x2))*0.8
box[:5] = 0
box[-5:] = 0
pl.plot(x,realdata,'k',lw=6)
pl.plot(x2,box,'k--',lw=6)

samples,w = importance(5000)
pl.hist(samples,normed=1,fc='k')
#pl.xlabel('x',fontsize=24)
#pl.ylabel('p(x)',fontsize=24)
pl.show()