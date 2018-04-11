#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 29 21:45:36 2017

@author: jabong
"""

import numpy as np
from sklearn.linear_model import ElasticNetCV
from norm import NormalizePositive
from sklearn import metrics

def predict(train):
    binary = (train > 0)
    reg = ElasticNetCV(fit_intercept=True, alphas=[
                       0.0125, 0.025, 0.05, .125, .25, .5, 1., 2., 4.])
    norm = NormalizePositive()
    train = norm.fit_transform(train)
    filled = train.copy()
    for u in range(filled.shape[0]):
        curtrain = np.delete(train, u, axis = 0)
        bu = binary[u]
        if np.sum(bu) > 5:
            reg.fit(curtrain[:, bu].T, train[u, bu])
            filled[u, ~bu] = reg.predict(curtrain[:, ~bu].T)
    return norm.inverse_transform(filled)
        
def main(transpose_inputs=False):
    from load_ml100k import get_test_train
    train,test = get_test_train(random_state=12)
    if transpose_inputs:
        train = train.T
        test = test.T
    filled = predict(train)
    r2 = metrics.r2_score(test[test > 0], filled[test > 0])

    print('R2 score ({} regression): {:.1%}'.format(
        ('movie' if transpose_inputs else 'user'),
        r2))

if __name__ == '__main__':
    main()
    main(transpose_inputs=True)