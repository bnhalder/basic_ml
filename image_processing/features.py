#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  1 15:02:53 2017

@author: jabong
"""

import numpy as np
import mahotas as mh

def edginess_sobel(im):
    edges = mh.sobel(im, just_filter=True)
    edges = edges.ravel()
    return np.sqrt(np.dot(edges, edges))

def texture(im):
    im = im.astype(np.uint8)
    return mh.features.haralick(im).ravel()

def chist(im):
    im //= 64
    r, g, b = im.transpose(2, 0, 1)
    pixels = 1*r + 4*g + 16*b
    hist = np.bincount(pixels.ravel(), minlength=64)
    hist = hist.astype(float)
    return np.log1p(hist)



