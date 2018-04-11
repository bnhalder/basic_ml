#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 21 18:01:27 2017

@author: jabong
"""

import numpy as np

def fit_model(k, features, labels):
    '''Learn a k-nn model'''
    # There is no model in k-nn, just a copy of the inputs
    return k, features.copy(), labels.copy()

def plurality(xs):
    '''find the most common element in a collection'''
    from collections import defaultdict
    counts = defaultdict(int)
    for x in xs:
        counts[x] += 1
    maxv = max(counts.values())
    for k, v in counts.items():
        if v == maxv:
            return k

def predict(features, model):
    '''Apply a k-nn model'''
    k, train_features, labels = model
    results = []
    for f in features:
        label_dist = []
        for t, ell in zip(train_features, labels):
            label_dist.append((np.linalg.norm(f-t), ell))
        label_dist.sort(key=lambda d_ell:d_ell[0])
        label_dist = label_dist[:k]
        results.append(plurality([ell for _,ell in label_dist]))
    return np.array(results)

def accuracy(features, labels, model):
    preds = predict(model, features)
    return np.mean(preds == labels)




        