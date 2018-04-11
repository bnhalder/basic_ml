#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 15 18:08:15 2017

@author: jabong
"""

import pylab as pl
import numpy as np

import pca
import kernelpca

data = np.zeros((150,2))

theta = np.random.normal(0,np.pi,50)
r = np.random.normal(0,0.1,50)
data[0:50,0] = r*np.cos(theta)
data[0:50,1] = r*np.sin(theta)

theta = np.random.normal(0,np.pi,50)
r = np.random.normal(2,0.1,50)
data[50:100,0] = r*np.cos(theta)
data[50:100,1] = r*np.sin(theta)

theta = np.random.normal(0,np.pi,50)
r = np.random.normal(5,0.1,50)
data[100:150,0] = r*np.cos(theta)
data[100:150,1] = r*np.sin(theta)

pl.figure()
pl.plot(data[:50,0],data[:50,1],'ok')
pl.plot(data[50:100,0],data[50:100,1],'^k')
pl.plot(data[100:150,0],data[100:150,1],'vk')
pl.title('Original dataset')

x,y,evals,evecs = pca.pca(data,2)
pl.figure()
pl.plot(x[:50,0],x[:50,1],'ok')
pl.plot(x[50:100,0],x[50:100,1],'^k')
pl.plot(x[100:150,0],x[100:150,1],'vk')
pl.title('Reconstructed points after PCA')

pl.figure()
y = kernelpca.kernelpca(data,'gaussian',2)
pl.plot(y[:50,0],y[:50,1],'ok')
pl.plot(y[50:100,0],y[50:100,1],'^k')
pl.plot(y[100:150,0],y[100:150,1],'vk')
pl.title('Reconstructed points after kernel PCA')

pl.show()