#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 19 19:54:43 2017

@author: jabong
"""

import numpy as np

def onemax(pop):
    
    fitness = np.sum(pop,axis=1)
        
    return fitness