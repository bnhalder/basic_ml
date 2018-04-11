#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 29 11:13:43 2017

@author: jabong
"""

def load():
    import numpy as np
    from scipy import sparse
    from os import path
    
    if not path.exists('data/ml-100k/u.data'):
        raise IOError('Please download data')
    
    data = np.loadtxt('data/ml-100k/u.data')
    ij = data[:, 0:2]
    ij -= 1
    values = data[:, 2]
    reviews = sparse.csc_matrix((values, ij.T)).astype(float)
    return reviews.toarray()

def get_test_train(reviews=None, random_state=None):
    import numpy as np
    import random
    r = random.Random(random_state)
    if reviews is None:
        reviews = load()
    U, M = np.where(reviews)
    test_indices = np.array(r.sample(range(len(U)), len(U)//10))
    
    train = reviews.copy()
    train[U[test_indices], M[test_indices]] = 0
    
    test = np.zeros_like(reviews)
    test[U[test_indices], M[test_indices]] = reviews[U[test_indices], M[test_indices]]
    return train, test  

train, test = get_test_train()