#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  1 14:57:04 2017

@author: jabong
"""

import mahotas as mh
import numpy as np

text = mh.imread("SimpleImageDataset/text21.jpg")
building = mh.imread("SimpleImageDataset/building00.jpg")
h, w, _ = text.shape
canvas = np.zeros((h, 2*w + 128, 3), np.uint8)
canvas[:, -w:] = building
canvas[:, :w] = text
canvas = canvas[::4, ::4]
mh.imsave('figure10.jpg', canvas)