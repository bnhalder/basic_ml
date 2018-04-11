#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Sep  3 12:54:12 2017

@author: jabong
"""

import os

DATA_DIR = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "data")

CHART_DIR = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "charts")

for d in [DATA_DIR, CHART_DIR]:
    if not os.path.exists(d):
        os.mkdir(d)
        
