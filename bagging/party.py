#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 22 16:41:13 2017

@author: jabong
"""

import numpy as np
#import dtree
import dtw
import bagging
import randomforest

tree = dtw.dtree()
#tree = dtree.dtree()
bagger = bagging.bagger()
forest = randomforest.randomforest()
party,classes,features = tree.read_data('party.data')

w = np.ones((np.shape(party)[0]),dtype = float)/np.shape(party)[0]

f = forest.rf(party,classes,features,10,7,2,maxlevel=2)
print "RF prediction"
print forest.rfclass(f,party)

t=tree.make_tree(party,w,classes,features)
print "Decision Tree prediction"
print tree.classifyAll(t,party)

print "Tree Stump Prediction"
print tree.classifyAll(t,party)

c=bagger.bag(party,classes,features,20)
print "Bagged Results"
print bagger.bagclass(c,party)

print "True Classes"
print classes
