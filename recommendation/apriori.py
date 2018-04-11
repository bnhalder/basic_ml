#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 31 12:58:01 2017

@author: jabong
"""

from collections import namedtuple

def apriori(dataset, minsupport, maxsize):
    from collections import defaultdict
    baskets = defaultdict(list)
    pointers = defaultdict(list)
    
    for i, ds in enumerate(dataset):
        for ell in ds:
            pointers[ell].append(i)
            baskets[frozenset([ell])].append(i)
    
    new_pointers = dict()
    for k in pointers:
        if len(pointers[k]) >= minsupport:
            new_pointers[k] = frozenset(pointers[k])
    pointers = new_pointers
    for k in baskets:
        baskets[k] = frozenset(baskets[k])
    
    valid = set()
    for el, c in baskets.items():
        if len(c) >= minsupport:
            valid.update(el)
    itemsets = [frozenset([v]) for v in valid]
    freqsets = []
    for i in range(maxsize-1):
        print("At iteration {}, number of frequent baskets: {}".format(i, len(itemsets)))
        newsets = []
        for it in itemsets:
            ccounts = baskets[it]
            for v,pv in pointers.items():
                if v not in it:
                    csup = (ccounts & pv)
                    if len(csup) >= minsupport:
                        new = frozenset(it | frozenset([v]))
                        if new not in baskets:
                            newsets.append(new)
                            baskets[new] = csup
        freqsets.extend(itemsets)
        itemsets = newsets
        if not len(itemsets):
            break
    support = {}
    for k in baskets:
        support[k] = float(len(baskets[k]))
    return freqsets, support

AssociationRule = namedtuple('AssociationRule', ['antecendent', 'consequent', 'base', 'py_x', 'lift'])

def association_rules(dataset, freqsets, support, minlift):
    nr_transactions = float(len(dataset))
    freqsets = [f for f in freqsets if len(f) > 1]
    for fset in freqsets:
        for f in fset:
            consequent = frozenset([f])
            antecendent = fset - consequent
            py_x = support[fset] / support[antecendent]
            base = support[consequent] / nr_transactions
            lift = py_x / base
            if lift > minlift:
                yield AssociationRule(antecendent, consequent, base, py_x, lift)