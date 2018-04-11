#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 19 19:56:06 2017

@author: jabong
"""

import pylab as pl
import numpy as np

#import fourpeaks as fF
import knapsack as fF

def PBIL():
    pl.ion()
    populationSize = 100
    stringLength = 20	
    eta = 0.005
    
    #fitnessFunction = 'fF.fourpeaks'
    fitnessFunction = 'fF.knapsack'
    p = 0.5*np.ones(stringLength)
    best = np.zeros(501,dtype=float)
    
    for count in range(501):
        population = np.random.rand(populationSize,stringLength)
        for i in range(stringLength):
            population[:,i] = np.where(population[:,i]<p[i],1,0)
        fitness = eval(fitnessFunction)(population)
        
        best[count] = np.max(fitness)
        bestplace = np.argmax(fitness)
        fitness[bestplace] = 0
        secondplace = np.argmax(fitness)
        
        p  = p*(1-eta) + eta*((population[bestplace,:]+population[secondplace,:])/2)
        if (np.mod(count,100)==0):
            print count, best[count]
        
    pl.plot(best,'kx-')
    pl.xlabel('Epochs')
    pl.ylabel('Fitness')
    pl.show()
    #print p

PBIL()
