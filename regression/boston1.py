#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 28 11:15:18 2017

@author: jabong
"""

import numpy as np
from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression
from matplotlib import pyplot as plt

boston = load_boston()
x = boston.data
y = boston.target

lr = LinearRegression()
lr.fit(x, y)

rmse = np.sqrt(lr.residues_/len(x))
print("rmse: {}".format(rmse))

fig, ax = plt.subplots()
#plot a diagonal for referece
ax.plot([0, 50], [0, 50], '-', color=(.9, .3, .3), lw=4)
ax.scatter(lr.predict(x), boston.target)
ax.set_xlabel('predicted')
ax.set_ylabel('real')
fig.savefig('Figure_07_08.png')



