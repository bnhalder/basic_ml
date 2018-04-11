#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 22 12:52:45 2017

@author: jabong
"""

import dtree

tree = dtree.dtree()
party,classes,features = tree.read_data('party.data')
t=tree.make_tree(party,classes,features)
tree.printTree(t,' ')

print tree.classifyAll(t,party)

for i in range(len(party)):
    tree.classify(t,party[i])

print "True Classes"
print classes