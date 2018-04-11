#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 22 12:50:59 2017

@author: jabong
"""
from __future__ import print_function
import numpy as np
from load import load_dataset
from sklearn.neighbors import KNeighborsClassifier

features, labels = load_dataset('seeds')
classifier = KNeighborsClassifier(n_neighbors = 4)

#leave-one-out cross validation
n = len(features)
error = 0.0
for el in range(n):
    training = np.ones(n, bool)
    training[el] = 0
    testing = ~training
    classifier.fit(features[training], labels[training])
    pred = classifier.predict(features[testing])
    error += (pred == labels[el])
print('result of leave-one-out: {}'.format(error/n))

from sklearn.cross_validation import KFold
kf = KFold(len(features), n_folds = 3, shuffle = True)
error = [] #error of each fold
for training, testing in kf:
    classifier.fit(features[training], labels[training])
    prediction = classifier.predict(features[testing])
    current_error = np.mean(prediction == labels[testing])
    error.append(current_error)
print('result of cross validation using kfold: {}'.format(error))

#below lines do the same things using library
from sklearn.cross_validation import cross_val_score
crossed = cross_val_score(classifier, features, labels)
print('result of cross-validation using cross_val_score: {}'.format(crossed))

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
classifier = Pipeline([('norm', StandardScaler()), ('knn', classifier)])
crossed = cross_val_score(classifier, features, labels)
print('result with prescaling: {}'.format(crossed))

#confusion matrix with cross-validation
from sklearn.metrics import confusion_matrix
names = sorted(set(labels))
labels = np.array([names.index(lab) for lab in labels])
preds = labels.copy()
preds[:] = -1
for train, test in kf:
    classifier.fit(features[train], labels[train])
    preds[test] = classifier.predict(features[test])

cmat = confusion_matrix(labels, preds)
print()
print('Confusion matrix: [rows represent true outcome, columns predicted outcome]')
print(cmat)

acc = cmat.trace()/float(cmat.sum())
print('Accuracy: {0:.2%}'.format(acc))

