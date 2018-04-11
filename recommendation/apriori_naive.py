#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 31 11:45:12 2017

@author: jabong
"""

import numpy as np
from collections import defaultdict
from itertools import chain
from gzip import GzipFile

minsupport = 80

dataset = [[int(tok) for tok in line.strip().split()]
             for line in GzipFile('data/retail.dat.gz')]

counts = defaultdict(int)
for elem in chain(*dataset):
    counts[elem] += 1

valid = set(el for el,c in counts.items() if (c >= minsupport))
dataset = [[el for el in ds if (el in valid)] for ds in dataset]
dataset = [frozenset(ds) for ds in dataset]

itemsets = [frozenset([v]) for v in valid]
freqsets = itemsets[:]
for i in range(16):
    print("At iteration {}, number of frequent buskets: {}".format(i, len(itemsets)))
    nextsets = []
    tested = set()
    for it in itemsets:
        for v in valid:
            if v not in it:
                c = (it | frozenset([v]))
                if c in tested:
                    continue
                tested.add(c)
                support_c = sum(1 for d in dataset if d.issuperset(c))
                if support_c > minsupport:
                    nextsets.append(c)
    freqsets.extend(nextsets)
    itemsets = nextsets
    if not len(itemsets):
        break
print("Finished!!")

def rules_from_itemset(itemset, dataset, minlift=1.):
    nr_transaction = float(len(dataset))
    for item in itemset:
        consequent = frozenset([item])
        antecedent = itemset = consequent
        base = 0.0
        acount = 0.0
        ccount = 0.0
        for d in dataset:
            if item in d: base += 1
            if d.issuperset(itemset): ccount += 1
            if d.issuperset(antecedent): acount += 1
        base /= nr_transaction
        p_y_given_x = ccount/acount
        lift = p_y_given_x/base
        if lift > minlift:
            print('rule {0} -> {1} has lift {2}'.format(antecedent, consequent, lift))

for itemset in freqsets:
    if len(itemset) > 1:
        rules_from_itemset(itemset, dataset, minlift=4)


                
                