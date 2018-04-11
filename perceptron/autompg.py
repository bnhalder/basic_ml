#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  7 15:50:31 2017

@author: jabong
"""

import numpy as np
import linreg

auto = np.genfromtxt('auto-mpg.data', comments='"')
auto = auto[~np.isnan(auto).any(axis=1)]

inputs = auto[:, 1:]
targets = auto[:, 0:1]

inputs -= inputs.mean(axis=0)
inputs /= inputs.std(axis=0)

#train = np.ones((inputs.shape[0], 1))
#train[::10] = 0
#train = np.argwhere(train)
#train_in = inputs[train[:, 0]]
#train_tgt = targets[train[:, 0]]

#test = np.zeros((inputs.shape[0], 1))
#test[::10] = 1
#test = np.argwhere(test)
#test_in = inputs[test[:, 0]]
#test_tgt = targets[test[:, 0]]

beta = linreg.linreg(inputs, targets)
test_in = np.concatenate((inputs,-np.ones((np.shape(inputs)[0],1))),axis=1)
testout = np.dot(test_in, beta)
error = np.sum((testout - targets)**2)
print error


