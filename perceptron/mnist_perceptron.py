#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  7 14:44:11 2017

@author: jabong
"""

import pylab as pl
import numpy as np
import pcn
import cPickle, gzip

f = gzip.open('mnist.pkl.gz', 'rb')
tset, vset, teset = cPickle.load(f)
f.close()

nread = 10000
train_in = tset[0][:nread, :]
train_tgt = np.zeros((nread, 10))
for i in range(nread):
    train_tgt[i, tset[1][i]] = 1

test_in = teset[0][:nread, :]
test_tgt = np.zeros((nread, 10))
for i in range(nread):
    test_tgt[i, teset[1][i]] = 1

p = pcn.pcn(train_in, train_tgt)
p.pcntrain(train_in, train_tgt, 0.25, 100)

p.confmat(train_in, train_tgt)

p.confmat(test_in, test_tgt)
