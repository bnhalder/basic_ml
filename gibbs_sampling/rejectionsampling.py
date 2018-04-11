#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 16:09:04 2017

@author: jabong
"""

import pylab as pl
import numpy as np

def qsample():
    return np.random.rand()*4

def p(x):
    return 0.3*np.exp(-(x-0.3)**2) + 0.7*np.exp(-(x-2.)**2/0.3)

def rejection(nsamples):
    M = 0.72
    samples = np.zeros(nsamples, dtype=float)
    count = 0
    for i in range(nsamples):
        accept = False
        while not accept:
            x = qsample()
            u = np.random.rand()*M
            if u<p(x):
                accept = True
                samples[i] = x
            else:
                count += 1
    print count
    return samples

x = np.arange(0,4,0.01)
x2 = np.arange(-0.5,4.5,0.1)
realdata = 0.3*np.exp(-(x-0.3)**2) + 0.7* np.exp(-(x-2.)**2/0.3)
box = np.ones(len(x2))*0.75#0.8
box[:5] = 0
box[-5:] = 0
pl.plot(x,realdata,'k',lw=6)
pl.plot(x2,box,'k--',lw=6)

import time
t0=time.time()
samples = rejection(10000)
t1=time.time()
print "Time ",t1-t0

pl.hist(samples,15,normed=1,fc='k')
pl.xlabel('x',fontsize=24)
pl.ylabel('p(x)',fontsize=24)
pl.axis([-0.5,4.5,0,1])
pl.show()