#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 22 18:01:32 2017

@author: jabong
"""

import pylab as pl
import numpy as np

import som
nNodesEdge = 8
data = (np.random.rand(2000,2)-0.5)*2

# Set up the network and decide on parameters
net = som.som(nNodesEdge,nNodesEdge,data,usePCA=0)
step = 0.2

pl.figure(1)
pl.plot(data[:,0],data[:,1],'.')
# Train the network for 0 iterations (to get the position of the nodes)
net.somtrain(data,0)
for i in range(net.x*net.y):
    neighbours = np.where(net.mapDist[i,:]<=step)

    t = np.zeros((np.shape(neighbours)[1]*2,np.shape(net.weights)[0]))
    t[::2,:] = np.tile(net.weights[:,i],(np.shape(neighbours)[1],1))
    t[1::2,:] = np.transpose(net.weights[:,neighbours[0][:]])
    pl.plot(t[:,0],t[:,1],'g-')
pl.axis('off')

pl.figure(2)
pl.plot(data[:,0],data[:,1],'.')
net.somtrain(data,5)
for i in range(net.x*net.y):
    neighbours = np.where(net.mapDist[i,:]<=step)

    t = np.zeros((np.shape(neighbours)[1]*2,np.shape(net.weights)[0]))
    t[::2,:] = np.tile(net.weights[:,i],(np.shape(neighbours)[1],1))
    t[1::2,:] = np.transpose(net.weights[:,neighbours[0][:]])
    pl.plot(t[:,0],t[:,1],'g-')
pl.axis([-1,1,-1,1])
pl.axis('off')

net.somtrain(data,100)
pl.figure(3)
pl.plot(data[:,0],data[:,1],'.')
for i in range(net.x*net.y):
    neighbours = np.where(net.mapDist[i,:]<=step)
    #print neighbours
    #n = tile(net.weights[:,i],(shape(neighbours)[1],1))
    t = np.zeros((np.shape(neighbours)[1]*2,np.shape(net.weights)[0]))
    t[::2,:] = np.tile(net.weights[:,i],(np.shape(neighbours)[1],1))
    t[1::2,:] = np.transpose(net.weights[:,neighbours[0][:]])
    pl.plot(t[:,0],t[:,1],'g-')
pl.axis([-1,1,-1,1])
pl.axis('off')

net.somtrain(data,100)
pl.figure(4)
pl.plot(data[:,0],data[:,1],'.')
for i in range(net.x*net.y):
    neighbours = np.where(net.mapDist[i,:]<=step)
    #print neighbours
    #n = np.tile(net.weights[:,i],(np.shape(neighbours)[1],1))
    t = np.zeros((np.shape(neighbours)[1]*2,np.shape(net.weights)[0]))
    t[::2,:] = np.tile(net.weights[:,i],(np.shape(neighbours)[1],1))
    t[1::2,:] = np.transpose(net.weights[:,neighbours[0][:]])
    pl.plot(t[:,0],t[:,1],'g-')
pl.axis([-1,1,-1,1])
pl.axis('off')
    
net.somtrain(data,100)
pl.figure(5)
pl.plot(data[:,0],data[:,1],'.')
for i in range(net.x*net.y):
    neighbours = np.where(net.mapDist[i,:]<=step)
    #print neighbours
    #n = np.tile(net.weights[:,i],(snp.hape(neighbours)[1],1))
    t = np.zeros((np.shape(neighbours)[1]*2,np.shape(net.weights)[0]))
    t[::2,:] = np.tile(net.weights[:,i],(np.shape(neighbours)[1],1))
    t[1::2,:] = np.transpose(net.weights[:,neighbours[0][:]])
    pl.plot(t[:,0],t[:,1],'g-')
pl.axis([-1,1,-1,1])
pl.axis('off')
pl.show()
    
    