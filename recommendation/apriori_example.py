#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 31 15:05:23 2017

@author: jabong
"""

from apriori import apriori, association_rules
from gzip import GzipFile

# Load dataset
dataset = [[int(tok) for tok in line.strip().split()]
           for line in GzipFile('data/retail.dat.gz')]

freqsets, support = apriori(dataset, 80, maxsize=16)
rules = list(association_rules(dataset, freqsets, support, minlift=30.0))

rules.sort(key=(lambda ar: -ar.lift))
for ar in rules:
    print('{} -> {} (lift = {:.4})'
          .format(set(ar.antecendent),
                    set(ar.consequent),
                    ar.lift))