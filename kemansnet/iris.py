#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 22 17:36:40 2017

@author: jabong
"""

import pylab as pl
import numpy as np

iris = np.loadtxt('../data/iris_proc.data',delimiter=',')
iris[:,:4] = iris[:,:4]-iris[:,:4].mean(axis=0)
imax = np.concatenate((iris.max(axis=0)*np.ones((1,5)),iris.min(axis=0)*np.ones((1,5))),axis=0).max(axis=0)
iris[:,:4] = iris[:,:4]/imax[:4]

target = iris[:,4]

order = range(np.shape(iris)[0])
np.random.shuffle(order)
iris = iris[order,:]
target = target[order]

train = iris[::2,0:4]
traint = target[::2]
valid = iris[1::4,0:4]
validt = target[1::4]
test = iris[3::4,0:4]
testt = target[3::4]

import kmeansnet
#import kmeans as kmeansnet
net = kmeansnet.kmeans(3,train)
net.kmeanstrain(train)
cluster = net.kmeansfwd(test)
print 1.*cluster
print iris[3::4,4]

import som
net = som.som(6,6,train)
net.somtrain(train,400)

best = np.zeros(np.shape(train)[0],dtype=int)
for i in range(np.shape(train)[0]):
    best[i],activation = net.somfwd(train[i,:])

pl.plot(net.map[0,:],net.map[1,:],'k.',ms=15)
where = pl.find(traint == 0)
pl.plot(net.map[0,best[where]],net.map[1,best[where]],'rs',ms=30)
where = pl.find(traint == 1)
pl.plot(net.map[0,best[where]],net.map[1,best[where]],'gv',ms=30)
where = pl.find(traint == 2)
pl.plot(net.map[0,best[where]],net.map[1,best[where]],'b^',ms=30)
pl.axis([-0.1,1.1,-0.1,1.1])
pl.axis('off')
pl.figure(2)

best = np.zeros(np.shape(test)[0],dtype=int)
for i in range(np.shape(test)[0]):
    best[i],activation = net.somfwd(test[i,:])

pl.plot(net.map[0,:],net.map[1,:],'k.',ms=15)
where = pl.find(testt == 0)
pl.plot(net.map[0,best[where]],net.map[1,best[where]],'rs',ms=30)
where = pl.find(testt == 1)
pl.plot(net.map[0,best[where]],net.map[1,best[where]],'gv',ms=30)
where = pl.find(testt == 2)
pl.plot(net.map[0,best[where]],net.map[1,best[where]],'b^',ms=30)
pl.axis([-0.1,1.1,-0.1,1.1])
pl.axis('off')
pl.show()