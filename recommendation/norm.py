#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 29 16:13:06 2017

@author: jabong
"""

import numpy as np

class NormalizePositive(object):
    
    def __init__(self, axis=0):
        self.axis = axis
    
    def fit(self, features, y=None):
        if self.axis == 1:
            features = features.T
        binary = (features > 0)
        count = binary.sum(axis=0)
        count[count==0] = 1
        self.mean = features.sum(axis=0)/count
        diff = (features - self.mean) * binary
        diff **= 2
        self.std = np.sqrt(1.0 + diff.sum(axis=0)/count)
        return self
    
    def transform(self, features):
        if self.axis == 1:
            features = features.T
        binary = (features > 0)
        features = features - self.mean
        features /= self.std
        features *= binary
        if self.axis == 1:
            features = features.T
        return features
    
    def inverse_transform(self, features, copy=True):
        if copy:
            features = features.copy()
        if self.axis == 1:
            features = features.T
        features *= self.std
        features += self.mean
        if self.axis == 1:
            features = features.T
        return features
    
    def fit_transform(self, features):
        return self.fit(features).transform(features)
    
def predict(train):
    norm = NormalizePositive()
    train = norm.fit_transform(train)
    return norm.inverse_transform(train * 0.)

def main(transpose_inputs=False):
    from load_ml100k import get_test_train
    from sklearn import metrics
    train,test = get_test_train(random_state=12)
    if transpose_inputs:
        train = train.T
        test = test.T
    predicted = predict(train)
    r2 = metrics.r2_score(test[test > 0], predicted[test > 0])
    print('R2 score ({} normalization): {:.1%}'.format(
        ('movie' if transpose_inputs else 'user'),
        r2))
    
if __name__ == '__main__':
    main()
    main(transpose_inputs=True)
    
    

