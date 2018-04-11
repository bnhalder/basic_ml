#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 31 11:21:32 2017

@author: jabong
"""

import numpy as np
from collections import defaultdict
from itertools import chain
from gzip import GzipFile

dataset = [[int(tok) for tok in line.strip().split()]
             for line in GzipFile('data/retail.dat.gz')]
counts = defaultdict(int)

for elem in chain(*dataset):
    counts[elem] += 1
count = np.array(list(counts.values()))
bins = [1, 2, 4, 8, 16, 32, 64, 128, 512]

print(' {0:11} | {1:12}'.format('Nr of baskets', 'Nr of products'))
print('--------------------------------')
for i in range(len(bins)):
    bot = bins[i]
    top = (bins[i+1] if (i+1)<len(bins) else 100000000000)
    print('   {0:4} - {1:3}    |  {2:12}'.format(
            bot, (top if top<1000 else ''), np.sum((count>=bot) & (count<top))))