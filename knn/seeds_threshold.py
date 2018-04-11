#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 22 11:51:00 2017

@author: jabong
"""

import numpy as np
from load import load_dataset
from threshold import fit_model, accuracy

features, labels = load_dataset('seeds')
labels = (labels == 'Canadian')

error = 0.0
for fold in range(10):
    training = np.ones(len(features), bool)
    #start from index fold, make every 10th element zero till last
    training[fold::10] = 0
    testing = ~training
    
    model = fit_model(features[training], labels[training])
    test_error = accuracy(features[testing], labels[testing], model)
    error += test_error
    
error /= 10.0
print('Ten fold cross-validation error was {0:.1%}.'.format(error))
    
