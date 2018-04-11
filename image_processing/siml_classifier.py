#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  1 15:19:47 2017

@author: jabong
"""

import mahotas as mh
import numpy as np
from glob import glob
from features import texture, chist
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

basedir = 'SimpleImageDataset/'
haralick, labels, chistogram = [], [], []

print('This script will test (with cross-validation) classification of the simple 3 class dataset')
print('Computing features...')
# Use glob to get all the images
images = glob('{}/*.jpg'.format(basedir))

for fname in sorted(images):
    imc = mh.imread(fname)
    haralick.append(texture(mh.colors.rgb2gray(imc)))
    chistogram.append(chist(imc))
    labels.append(fname[:-len('xx.jpg')])

print('Finished computing features.')

haralick = np.array(haralick)
chistogram = np.array(chistogram)
labels = np.array(labels)

haralick_plus_chist = np.hstack([chistogram, haralick])

clf = Pipeline([('preproc', StandardScaler()), ('classifier', LogisticRegression())])

from sklearn import cross_validation
cv = cross_validation.LeaveOneOut(len(images))
scores = cross_validation.cross_val_score(clf, haralick, labels, cv=cv)
print('Accuracy (Leave-one-out) with Logistic Regression [haralick features]: {:.1%}'.format(scores.mean()))

scores = cross_validation.cross_val_score(clf, chistogram, labels, cv=cv)
print('Accuracy (Leave-one-out) with Logistic Regression [color histograms]: {:.1%}'.format(
    scores.mean()))

scores = cross_validation.cross_val_score(clf, haralick_plus_chist, labels, cv=cv)
print('Accuracy (Leave-one-out) with Logistic Regression [texture features + color histograms]: {:.1%}'.format(scores.mean()))
