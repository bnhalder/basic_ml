#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 19 17:11:37 2017

@author: jabong
"""

import numpy as np

def makeTSP(nCities):
    positions = 2*np.random.rand(nCities, 2) - 1
    distances = np.zeros((nCities,nCities))
    
    for i in range(nCities):
        for j in range(i+1, nCities):
            distances[i,j] = np.sqrt((positions[i,0] - positions[j,0])**2 + (positions[i,1] - positions[j,1])**2);
            distances[j,i] = distances[i,j];
    
    return distances

def exhaustive(distances):
    nCities = np.shape(distances)[0]
    cityOrder = np.arange(nCities)
    
    distanceTravelled = 0
    for i in range(nCities-1):
        distanceTravelled += distances[cityOrder[i],cityOrder[i+1]]
    distanceTravelled += distances[cityOrder[nCities-1],0]
    
    for newOrder in permutation(range(nCities)):
        possibleDistanceTravelled = 0
        for i in range(nCities-1):
            possibleDistanceTravelled += distances[newOrder[i],newOrder[i+1]]
        possibleDistanceTravelled += distances[newOrder[nCities-1],0]
        
        if possibleDistanceTravelled < distanceTravelled:
            distanceTravelled = possibleDistanceTravelled
            cityOrder = newOrder
    return cityOrder, distanceTravelled

def permutation(order):
    order = tuple(order)
    if len(order) == 1:
        yield order
    else:
        for i in range(len(order)):
            rest = order[:i] + order[i+1:]
            move = (order[i],)
            for smaller in permutation(rest):
                yield move + smaller

def greedy(distances):
    nCities = np.shape(distances)[0]
    distanceTravelled = 0
    
    dist = distances.copy()
    cityOrder = np.zeros(nCities)
    cityOrder[0] = np.random.randint(nCities)
    dist[:, int(cityOrder[0])] = np.Inf
    
    for i in range(nCities-1):
        cityOrder[i+1] = np.argmin(dist[int(cityOrder[i]),:])
        distanceTravelled  += dist[int(cityOrder[i]),int(cityOrder[i+1])]
        dist[:,int(cityOrder[i+1])] = np.Inf
    
    distanceTravelled += distances[int(cityOrder[nCities-1]),0]
    return cityOrder, distanceTravelled

def hillClimbing(distances):
    nCities = np.shape(distances)[0]
    cityOrder = np.arange(nCities)
    np.random.shuffle(cityOrder)
    
    distanceTravelled = 0
    for i in range(nCities-1):
        distanceTravelled += distances[cityOrder[i],cityOrder[i+1]]
    distanceTravelled += distances[cityOrder[nCities-1],0]
    
    for i in range(1000):
        city1 = np.random.randint(nCities)
        city2 = np.random.randint(nCities)
        
        if city1 != city2:
            possibleCityOrder = cityOrder.copy()
            possibleCityOrder = np.where(possibleCityOrder==city1,-1,possibleCityOrder)
            possibleCityOrder = np.where(possibleCityOrder==city2,city1,possibleCityOrder)
            possibleCityOrder = np.where(possibleCityOrder==-1,city2,possibleCityOrder)
            
            newDistanceTravelled = 0
            for j in range(nCities-1):
                newDistanceTravelled += distances[possibleCityOrder[j],possibleCityOrder[j+1]]
            distanceTravelled += distances[cityOrder[nCities-1],0]
            
            if newDistanceTravelled < distanceTravelled:
                distanceTravelled = newDistanceTravelled
                cityOrder = possibleCityOrder
    return cityOrder, distanceTravelled

def simulatedAnnealing(distances):
    nCities = np.shape(distances)[0]
    cityOrder = np.arange(nCities)
    np.random.shuffle(cityOrder)
    
    distanceTravelled = 0
    for i in range(nCities-1):
        distanceTravelled += distances[cityOrder[i],cityOrder[i+1]]
    distanceTravelled += distances[cityOrder[nCities-1],0]
    
    T = 500
    c = 0.8
    nTests = 10
    
    while T>1:
        for i in range(nTests):
            city1 = np.random.randint(nCities)
            city2 = np.random.randint(nCities)
            
            if city1 != city2:
                possibleCityOrder = cityOrder.copy()
                possibleCityOrder = np.where(possibleCityOrder==city1,-1,possibleCityOrder)
                possibleCityOrder = np.where(possibleCityOrder==city2,city1,possibleCityOrder)
                possibleCityOrder = np.where(possibleCityOrder==-1,city2,possibleCityOrder)
                
                newDistanceTravelled = 0
                for j in range(nCities-1):
                    newDistanceTravelled += distances[possibleCityOrder[j],possibleCityOrder[j+1]]
                distanceTravelled += distances[cityOrder[nCities-1],0]
                
                if newDistanceTravelled < distanceTravelled or (distanceTravelled - newDistanceTravelled) < T*np.log(np.random.rand()):
                    distanceTravelled = newDistanceTravelled
                    cityOrder = possibleCityOrder
            T = c*T
    return cityOrder, distanceTravelled

def runAll():
	import time

	nCities = 10
	distances = makeTSP(nCities)

	print "Exhaustive search"
	start = time.time()
	print exhaustive(distances)
	finish = time.time()
	print finish-start

	print "Greedy search"
	start = time.time()
	print greedy(distances)
	finish = time.time()
	print finish-start

	print "Hill Climbing"
	start = time.time()
	print hillClimbing(distances)
	finish = time.time()
	print finish-start

	print "Simulated Annealing"
	start = time.time()
	print simulatedAnnealing(distances)
	finish = time.time()
	print finish-start

runAll()
    
        
    