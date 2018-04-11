#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 11 16:45:59 2017

@author: jabong
"""

import pylab as pl
import numpy as np

x = np.linspace(0, 1, 40).reshape((40, 1))
t = np.sin(2*np.pi*x) + np.cos(4*np.pi*x) + np.random.randn(40).reshape((40, 1))*0.2
x = (x - 0.5) * 2

train = x[0::2, :]
test = x[1::4, :]
valid = x[3::4, :]
traintarget = t[0::2,:]
testtarget = t[1::4,:]
validtarget = t[3::4,:]

pl.plot(x, t, 'o')
pl.xlabel('x')
pl.ylabel('t')

import mlp
net = mlp.mlp(train, traintarget, 3, outtype='linear')
net.mlptrain(train, traintarget, 0.25, 101)

net.earlystopping(train,traintarget,valid,validtarget,0.25)

# Test out different sizes of network
count = 0
out = np.zeros((10,7))
for nnodes in [1,2,3,5,10,25,50]:
    for i in range(10):
        net = mlp.mlp(train,traintarget,nnodes,outtype='linear')
        out[i,count] = net.earlystopping(train,traintarget,valid,validtarget,0.25)
    count += 1
    
test = np.concatenate((test,-np.ones((np.shape(test)[0],1))),axis=1)
outputs = net.mlpfwd(test)
print 0.5*sum((outputs-testtarget)**2)

print out
print out.mean(axis=0)
print out.var(axis=0)
print out.max(axis=0)
print out.min(axis=0)

pl.show()