#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  7 15:15:30 2017

@author: jabong
"""

import numpy as np
import pylab as pl
import pcn

pima_data = np.loadtxt('pima-indians-diabetes.data', delimiter = ',')

indices0 = np.where(pima_data[:,8]==0)
indices1 = np.where(pima_data[:,8]==1)

#pl.ion()
#pl.plot(pima_data[indices0,0], pima_data[indices0,1],'go')
#pl.plot(pima_data[indices1,0], pima_data[indices1,1],'rx')

# Perceptron training on the original dataset
print "Output on original data"
p = pcn.pcn(pima_data[:,:8], pima_data[:,8:9])
p.pcntrain(pima_data[:,:8], pima_data[:,8:9],0.25,100)
p.confmat(pima_data[:,:8], pima_data[:,8:9])

# Various preprocessing steps
pima_data[np.where(pima_data[:,0]>8),0] = 8

pima_data[np.where(pima_data[:,7]<=30),7] = 1
pima_data[np.where((pima_data[:,7]>30) & (pima_data[:,7]<=40)),7] = 2
pima_data[np.where((pima_data[:,7]>40) & (pima_data[:,7]<=50)),7] = 3
pima_data[np.where((pima_data[:,7]>50) & (pima_data[:,7]<=60)),7] = 4
pima_data[np.where(pima_data[:,7]>60),7] = 5

pima_data[:,:8] = pima_data[:,:8]-pima_data[:,:8].mean(axis=0)
pima_data[:,:8] = pima_data[:,:8]/pima_data[:,:8].var(axis=0)

trainin = pima_data[::2,:8]
testin = pima_data[1::2,:8]
traintgt = pima_data[::2,8:9]
testtgt = pima_data[1::2,8:9]

# Perceptron training on the preprocessed dataset
print "Output after preprocessing of data"
p1 = pcn.pcn(trainin,traintgt)
p1.pcntrain(trainin,traintgt,0.25,100)
p1.confmat(testin,testtgt)
