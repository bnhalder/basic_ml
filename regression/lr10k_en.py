#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 28 20:30:53 2017

@author: jabong
"""

import numpy as np
from sklearn.datasets import load_svmlight_file
from sklearn.cross_validation import KFold
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import ElasticNet, ElasticNetCV
from matplotlib import pyplot as plt

data, target = load_svmlight_file('data/E2006.train')
met = ElasticNet(alpha=0.1)

kf = KFold(len(target), n_folds=5)
pred = np.zeros_like(target)
for train, test in kf:
    met.fit (data[train], target[train])
    pred[test] = met.predict(data[test])

print('[EN 0.1] RMSE on testing (5-fold), {:.2}'.format(np.sqrt(mean_squared_error(target, pred))))
print('[EN 0.1] R2 on testing (5-fold), {:.2}'.format(r2_score(target, pred)))
print('')

met = ElasticNetCV(n_jobs = -1)
kf = KFold(len(target), n_folds=5)
pred = np.zeros_like(target)
for train, test in kf:
    met.fit (data[train], target[train])
    pred[test] = met.predict(data[test])

print('[EN CV] RMSE on testing (5-fold), {:.2}'.format(np.sqrt(mean_squared_error(target, pred))))
print('[EN CV] R2 on testing (5-fold), {:.2}'.format(r2_score(target, pred)))
print('')

met.fit(data, target)
pred = met.predict(data)
print('[EN CV] RMSE on training, {:.2}'.format(np.sqrt(mean_squared_error(target, pred))))
print('[EN CV] R2 on training, {:.2}'.format(r2_score(target, pred)))

# Construct an ElasticNetCV object (use all available CPUs)
met = ElasticNetCV(n_jobs=-1, l1_ratio=[.01, .05, .25, .5, .75, .95, .99])

kf = KFold(len(target), n_folds=5)
pred = np.zeros_like(target)
for train, test in kf:
    met.fit(data[train], target[train])
    pred[test] = met.predict(data[test])


print('[EN CV l1_ratio] RMSE on testing (5 fold), {:.2}'.format(np.sqrt(mean_squared_error(target, pred))))
print('[EN CV l1_ratio] R2 on testing (5 fold), {:.2}'.format(r2_score(target, pred)))
print('')


fig, ax = plt.subplots()
y = target
ax.scatter(y, pred, c='k')
ax.plot([-5,-1], [-5,-1], 'r-', lw=2)
ax.set_xlabel('Actual value')
ax.set_ylabel('Predicted value')
fig.savefig('Figure_10k_scatter_EN_l1_ratio.png')


