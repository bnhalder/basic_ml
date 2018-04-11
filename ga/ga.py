#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 19 20:03:26 2017

@author: jabong
"""

import pylab as pl
import numpy as np
import fourpeaks as fF

class ga:
    def __init__(self, stringLength, fitnessFunction, nEpochs, populationSize=100, mutationProb=-1, crossover='un', nElite=4, tournament=True):
        self.stringLength = stringLength
        
        if np.mod(populationSize,2)==0:
            self.populationSize = populationSize
        else:
            self.populationSize = populationSize+1
        
        if mutationProb < 0:
            self.mutationProb = 1/stringLength
        else:
            self.mutationProb = mutationProb
        
        self.nEpochs = nEpochs
        self.fitnessFunction = fitnessFunction
        
        self.crossover = crossover
        self.nElite = nElite
        self.tournment = tournament
        
        self.population = np.random.rand(self.populationSize,self.stringLength)
        self.population = np.where(self.population<0.5,0,1)
    
    def run_ga(self, plotfig):
        pl.ion()
        bestfit = np.zeros(self.nEpochs)
        
        for i in range(self.nEpochs):
            fitness = eval(self.fitnessFunction)(self.population)
            newPopulation = self.fps(self.population,fitness)
            
            if self.crossover == 'sp':
                newPopulation = self.spCrossover(newPopulation)
            elif self.crossover == 'un':
                newPopulation = self.uniformCrossover(newPopulation)
            newPopulation = self.mutate(newPopulation)
            
            if self.nElite>0:
                newPopulation = self.elitism(self.population,newPopulation,fitness)
            
            if self.tournament:
                newPopulation = self.tournament(self.population,newPopulation,fitness,self.fitnessFunction)
            
            self.population = newPopulation
            bestfit[i] = fitness.max()
            
            if (np.mod(i,100)==0):
                print i, fitness.max()
            #pl.plot([i],[fitness.max()],'r+')
        pl.plot(bestfit,'kx-')
    
    def fps(self,population,fitness):
        fitness = fitness/np.sum(fitness)
        fitness = 10*fitness/fitness.max()
        
        j = 0
        while np.round(fitness[j])<1:
            j = j + 1
        
        newPopulation = np.kron(np.ones((int(np.round(fitness[j])),1)),population[j,:])
        for i in range(j+1,self.populationSize):
            if np.round(fitness[i])>=1:
                newPopulation = np.concatenate((newPopulation,np.kron(np.ones((int(np.round(fitness[i])),1)),population[i,:])),axis=0)
        
        indices = range(np.shape(newPopulation)[0])
        np.random.shuffle(indices)
        newPopulation = newPopulation[indices[:self.populationSize],:]
        return newPopulation
    
    def spCrossover(self,population):
        newPopulation = np.zeros(np.shape(population))
        crossoverPoint = np.random.randint(0,self.stringLength,self.populationSize)
        for i in range(0,self.populationSize,2):
            newPopulation[i,:crossoverPoint[i]] = population[i,:crossoverPoint[i]]
            newPopulation[i+1,:crossoverPoint[i]] = population[i+1,:crossoverPoint[i]]
            newPopulation[i,crossoverPoint[i]:] = population[i+1,crossoverPoint[i]:]
            newPopulation[i+1,crossoverPoint[i]:] = population[i,crossoverPoint[i]:]
        return newPopulation
    
    def uniformCrossover(self,population):
        newPopulation = np.zeros(np.shape(population))
        which = np.random.rand(self.populationSize,self.stringLength)
        which1 = which>=0.5
        for i in range(0,self.populationSize,2):
            newPopulation[i,:] = population[i,:]*which1[i,:] + population[i+1,:]*(1-which1[i,:])
            newPopulation[i+1,:] = population[i,:]*(1-which1[i,:]) + population[i+1,:]*which1[i,:]
        return newPopulation
    
    def mutate(self,population):
        whereMutate = np.random.rand(np.shape(population)[0],np.shape(population)[1])
        population[np.where(whereMutate < self.mutationProb)] = 1 - population[np.where(whereMutate < self.mutationProb)]
        return population
    
    def elitism(self,oldPopulation,population,fitness):
        best = np.argsort(fitness)
        best = np.squeeze(oldPopulation[best[-self.nElite:],:])
        indices = range(np.shape(population)[0])
        np.random.shuffle(indices)
        population = population[indices,:]
        population[0:self.nElite,:] = best
        return population
    
    def tournament(self,oldPopulation,population,fitness,fitnessFunction):
        newFitness = eval(self.fitnessFunction)(population)
        for i in range(0,np.shape(population)[0],2):
            f = np.concatenate((fitness[i:i+2],newFitness[i:i+2]),axis=0)
            indices = np.argsort(f)
            if indices[-1]<2 and indices[-2]<2:
                population[i,:] = oldPopulation[i,:]
                population[i+1,:] = oldPopulation[i+1,:]
            elif indices[-1]<2:
                if indices[0]>=2:
                    population[i+indices[0]-2,:] = oldPopulation[i+indices[-1]]
                else:
                    population[i+indices[1]-2,:] = oldPopulation[i+indices[-1]]
            elif indices[-2]<2:
                if indices[0]>=2:
                    population[i+indices[0]-2,:] = oldPopulation[i+indices[-2]]
                else:
                    population[i+indices[1]-2,:] = oldPopulation[i+indices[-2]]
        return population
        
        
                    