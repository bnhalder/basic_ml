#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  1 16:39:05 2017

@author: jabong
"""

from sklearn.cluster import KMeans
from mahotas.features import surf

print('Computing SURF descriptors...')
alldescriptors = []
for im, _ in images():
    im = mh.imread(im, as_gray=True)
    im = im.astype(np.uint8)
    
    # To use dense sampling, you can try the following line:
    # alldescriptors.append(surf.dense(im, spacing=16))
    alldescriptors.append(surf.surf(im, descriptor_only=True))
    
print('Descriptor computation complete.')

k=256
km = KMeans(k)

concatenated = np.concatenate(alldescriptors)
print('Number of descriptors: {}'.format(len(concatenated)))
concatenated = concatenated[::64]
print('Clustering with K-means...')
km.fit(concatenated)
sfeatures = []
for d in alldescriptors:
    c = km.predict(d)
    sfeatures.append(np.bincount(c, minlength=k))
sfeatures = np.array(sfeatures, dtype=float)
print('predicting...')
score_SURF = cross_validation.cross_val_score(clf, sfeatures, labels, cv=cv).mean()
print('Accuracy (5 fold x-val) with Logistic Regression [SURF features]: {:.1%}'.format(
    score_SURF.mean()))

print('Performing classification with all features combined...')
allfeatures = np.hstack([sfeatures, ifeatures])
score_SURF_global = cross_validation.cross_val_score(clf, allfeatures, labels, cv=cv).mean()
print('Accuracy (5 fold x-val) with Logistic Regression [All features]: {:.1%}'.format(
    score_SURF_global.mean())) 