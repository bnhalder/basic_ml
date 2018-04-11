#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 11 17:15:04 2017

@author: jabong
"""

import pylab as pl
import numpy as np

PNoz = np.loadtxt('PNoz.dat')
pl.ion()
pl.plot(np.arange(np.shape(PNoz)[0]), PNoz[:,2], '.')
pl.xlabel('Time (Days)')
pl.ylabel('Ozone (Dobson units)')

PNoz[:,2] = PNoz[:,2]-PNoz[:,2].mean()
PNoz[:,2] = PNoz[:,2]/PNoz[:,2].max()

t=2
k=3

lastPoint = np.shape(PNoz)[0]-t*(k+1)
inputs = np.zeros((lastPoint,k))
targets = np.zeros((lastPoint,1))
for i in range(lastPoint):
    inputs[i,:] = PNoz[i:i+t*k:t,2]
    targets[i] = PNoz[i+t*(k+1),2]
    
test = inputs[-400:,:]
testtargets = targets[-400:]
train = inputs[:-400:2,:]
traintargets = targets[:-400:2]
valid = inputs[1:-400:2,:]
validtargets = targets[1:-400:2]

change = range(np.shape(inputs)[0])
np.random.shuffle(change)
inputs = inputs[change,:]
targets = targets[change,:]

import mlp
net = mlp.mlp(train,traintargets,3,outtype='linear')
net.earlystopping(train,traintargets,valid,validtargets,0.25)

test = np.concatenate((test,-np.ones((np.shape(test)[0],1))),axis=1)
testout = net.mlpfwd(test)

pl.figure()
pl.plot(np.arange(np.shape(test)[0]),testout,'.')
pl.plot(np.arange(np.shape(test)[0]),testtargets,'x')
pl.legend(('Predictions','Targets'))
print 0.5*np.sum((testtargets-testout)**2)
pl.show()