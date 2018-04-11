#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 22 16:47:28 2017

@author: jabong
"""

import numpy as np
import dtree
import bagging
import randomforest

tree = dtree.dtree()
bagger = bagging.bagger()
forest = randomforest.randomforest()

data,classes,features = tree.read_data('car.data')

train = data[::2][:]
test = data[1::2][:]
trainc = classes[::2]
testc = classes[1::2]

t=tree.make_tree(train,trainc,features)
out = tree.classifyAll(t,test)
tree.printTree(t,' ')
 
a = np.zeros(len(out))
b = np.zeros(len(out))
d = np.zeros(len(out))

for i in range(len(out)):
    if testc[i] == 'good' or testc[i] == 'v-good':
        b[i] = 1
        if out[i] == testc[i]:
            d[i] = 1
    if out[i] == testc[i]:
        a[i] = 1

print "Tree"
print "Number correctly predicted",np.sum(a)
print "Number of testpoints ",len(a)
print "Percentage Accuracy ",np.sum(a)/len(a)*100.0
print ""
print "Number of cars rated as good or very good", np.sum(b)
print "Number correctly identified as good or very good",np.sum(d) 
print "Percentage Accuracy",np.sum(d)/np.sum(b)*100.0

c=bagger.bag(train,trainc,features,100)
out = bagger.bagclass(c,test)
 
a = np.zeros(len(out))
b = np.zeros(len(out))
d = np.zeros(len(out))
 
for i in range(len(out)):
    if testc[i] == 'good' or testc[i]== 'v-good':
        b[i] = 1
        if out[i] == testc[i]:
            d[i] = 1
    if out[i] == testc[i]:
        a[i] = 1
print "-----"
print "Bagger"
print "Number correctly predicted",np.sum(a)
print "Number of testpoints ",len(a)
print "Percentage Accuracy ",np.sum(a)/len(a)*100.0
print ""
print "Number of cars rated as good or very good", np.sum(b)
print "Number correctly identified as good or very good",np.sum(d) 
print "Percentage Accuracy",np.sum(d)/np.sum(b)*100.0

f=f = forest.rf(train,trainc,features,100,200,2)
out = forest.rfclass(f,test)

a = np.zeros(len(out))
b = np.zeros(len(out))
d = np.zeros(len(out))

for i in range(len(out)):
    if testc[i] == 'good' or testc[i]== 'v-good':
        b[i] = 1
        if out[i] == testc[i]:
            d[i] = 1
    if out[i] == testc[i]:
        a[i] = 1
print "-----"
print "Forest"
print "Number correctly predicted",np.sum(a)
print "Number of testpoints ",len(a)
print "Percentage Accuracy ",np.sum(a)/len(a)*100.0
print ""
print "Number of cars rated as good or very good", np.sum(b)
print "Number correctly identified as good or very good",np.sum(d) 
print "Percentage Accuracy",np.sum(d)/np.sum(b)*100.0
