#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 22 13:38:34 2017

@author: jabong
"""

from __future__ import print_function
from matplotlib import pyplot as plt
from load import load_dataset
import numpy as np

from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

features, labels = load_dataset('seeds')

#values of K (no of neighbors) to consider: all in 1 to 160
ks = np.arange(1, 161)

classifier = KNeighborsClassifier()
classifier = Pipeline([('norm', StandardScaler()), ('knn', classifier)])

accuracies = []
for k in ks:
    classifier.set_params(knn__n_neighbors = k)
    crossed = cross_val_score(classifier, features, labels)
    accuracies.append(crossed.mean())
    
accuracies = np.array(accuracies)

plt.plot(ks, accuracies*100)
plt.xlabel('Value for k (no of neighbors)')
plt.ylabel('Accuracy in %')
plt.tight_layout()
plt.savefig('figure6.png')