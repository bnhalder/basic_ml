#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 11 16:59:29 2017

@author: jabong
"""

import numpy as np

iris = np.loadtxt('iris_proc.data', delimiter=',')
iris[:, :4] = iris[:, :4] - iris[:, :4].mean(axis=0)
imax = np.concatenate((iris.max(axis=0)*np.ones((1,5)), np.abs(iris.min(axis=0)*np.ones((1,5)))), axis=0).max(axis=0)
iris[:,:4] = iris[:,:4]/imax[:4]
print iris[0:5,:]

target = np.zeros((np.shape(iris)[0], 3))
indices = np.where(iris[:, 4] == 0)
target[indices, 0] = 1
indices = np.where(iris[:, 4] == 1)
target[indices, 1] = 1
indices = np.where(iris[:, 4] == 2)
target[indices, 2] = 1

order = range(np.shape(iris)[0])
np.random.shuffle(order)
iris = iris[order, :]
target = target[order, :]
train = iris[::2,0:4]
traint = target[::2]
valid = iris[1::4,0:4]
validt = target[1::4]
test = iris[3::4,0:4]
testt = target[3::4]

import mlp
net = mlp.mlp(train, traint, 5, outtype = 'logistic')
net.earlystopping(train, traint, valid, validt, 0.1)
net.confmat(test, testt)
