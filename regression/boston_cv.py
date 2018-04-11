#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 28 17:36:01 2017

@author: jabong
"""

from __future__ import print_function
import numpy as np
from sklearn.cross_validation import KFold
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.metrics import r2_score
from sklearn.datasets import load_boston

boston = load_boston()
x = boston.data
y = boston.target

for name, met in [
        ('linear regression', LinearRegression()),
        ('lasso()', Lasso()),
        ('elastic-net(.5)', ElasticNet(alpha=0.5)),
        ('lasso(.5)', Lasso(alpha=0.5)),
        ('ridge(.5)', Ridge(alpha=0.5))]:
    met.fit(x, y)
    p = met.predict(x)
    r2_train = r2_score(y, p)
    
    kf = KFold(len(x), n_folds=5)
    p = np.zeros_like(y)
    for train, test in kf:
        met.fit(x[train], y[train])
        p[test] = met.predict(x[test])
    r2_cv = r2_score(y, p)
    print('Method: {}'.format(name))
    print('R2 on training: {}'.format(r2_train))
    print('R2 on 5-fold CV: {}'.format(r2_cv))
    print()
    print()
        
    
    
