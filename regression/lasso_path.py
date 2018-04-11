#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 28 20:16:28 2017

@author: jabong
"""

from sklearn.linear_model import Lasso
from sklearn.datasets import load_boston
from matplotlib import pyplot as plt
import numpy as np

boston = load_boston()
x = boston.data
y = boston.target

las = Lasso(normalize=1)
alphas = np.logspace(-5, 2, 1000)
alphas, coefs, _ = las.path(x, y, alphas=alphas)

fig,ax = plt.subplots()
ax.plot(alphas, coefs.T)
ax.set_xscale('log')
ax.set_xlim(alphas.max(), alphas.min())
ax.set_xlabel('Lasso coefficient path as a function of alpha')
ax.set_xlabel('Alpha')
ax.set_ylabel('Coefficient weight')
fig.savefig('Figure_LassoPath.png')