#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 15 18:13:32 2017

@author: jabong
"""

import pylab as pl
import numpy as np

ecoli = np.loadtxt('../data/shortecoli.dat')
labels = ecoli[:,7:]
data = ecoli[:,:7]
data -= np.mean(data,axis=0)
data /= data.max(axis=0)

order = range(np.shape(data)[0])
np.random.shuffle(order)
data = data[order]
w0 = np.where(labels==1)
w1 = np.where(labels==2)
w2 = np.where(labels==3)

import lda
newData,w = lda.lda(data,labels,2)

pl.plot(data[w0,0],data[w0,1],'ok')
pl.plot(data[w1,0],data[w1,1],'^k')
pl.plot(data[w2,0],data[w2,1],'vk')
pl.axis([-1.5,1.8,-1.5,1.8])
pl.axis('off')
pl.figure(2)
pl.plot(newData[w0,0],newData[w0,1],'ok')
pl.plot(newData[w1,0],newData[w1,1],'^k')
pl.plot(newData[w2,0],newData[w2,1],'vk')
pl.axis([-1.5,1.8,-1.5,1.8])
pl.axis('off')

import pca
x,y,evals,evecs = pca.pca(data,2)
pl.figure(3)
pl.plot(y[w0,0],y[w0,1],'ok')
pl.plot(y[w1,0],y[w1,1],'^k')
pl.plot(y[w2,0],y[w2,1],'vk')
pl.axis('off')

import kernelpca
newData = kernelpca.kernelpca(data,'gaussian',2)
pl.figure(4)
pl.plot(newData[w0,0],newData[w0,1],'ok')
pl.plot(newData[w1,0],newData[w1,1],'^k')
pl.plot(newData[w2,0],newData[w2,1],'vk')
pl.axis('off')

pl.show()