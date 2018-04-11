#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 28 17:17:59 2017

@author: jabong
"""

import numpy as np
from sklearn.datasets import load_boston
import pylab as plt

boston = load_boston()
x = np.array([np.concatenate((v, [1])) for v in boston.data])
y = boston.target

# np.linal.lstsq implements least-squares linear regression
s, total_error, _, _ = np.linalg.lstsq(x, y)
rmse = np.sqrt(total_error[0]/len(x))
print('Residual: {}'.format(rmse))

plt.plot(np.dot(x, s), boston.target, 'ro')
plt.plot([0, 50], [0, 50], 'g-')
plt.plot([0, 50], [0, 50], 'g-')
plt.xlabel('predicted')
plt.ylabel('real')
plt.show()
