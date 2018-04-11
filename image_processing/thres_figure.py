#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  1 14:26:45 2017

@author: jabong
"""

import mahotas as mh
import numpy as np
from matplotlib import pyplot as plt

image = mh.imread('SimpleImageDataset/scene00.jpg')
image = mh.colors.rgb2grey(image, dtype=np.uint8)
plt.imshow(image)
plt.gray()
plt.title('original image')

thresh = mh.thresholding.otsu(image)
print('Otsu threshold is {}.'.format(thresh))

threshold = (image>thresh)
plt.figure()
plt.imshow(threshold)
plt.title('threholded image')
mh.imsave('thresholded.png', threshold.astype(np.uint8)*255)

im16 = mh.gaussian_filter(image, 16)
thresh = mh.thresholding.otsu(im16.astype(np.uint8))
threshed = (im16 > thresh)
plt.figure()
plt.imshow(threshed)
plt.title('threholded image (after blurring)')
print('Otsu threshold after blurring is {}.'.format(thresh))
mh.imsave('thresholded16.png', threshed.astype(np.uint8)*255)
plt.show()

image = mh.imread('SimpleImageDataset/building05.jpg')
image = mh.colors.rgb2grey(image, dtype=np.uint8)

th = mh.thresholding.otsu(image)
print('Otsu threshold is {0}'.format(thresh))

otsubin = (image > thresh)
print('Saving thresholded image (with Otsu threshold) to otsu-threshold.jpeg')
mh.imsave('otsu-threshold.jpeg', otsubin.astype(np.uint8) * 255)

otsubin = mh.open(otsubin, np.ones((15, 15)))
mh.imsave('otsu-closed.jpeg', otsubin.astype(np.uint8) * 255)

th = mh.thresholding.rc(image)
print('Ridley-Calvard threshold is {0}'.format(thresh))
print('Saving thresholded image (with Ridley-Calvard threshold) to rc-threshold.jpeg')
mh.imsave('rc-threshold.jpeg', (image > thresh).astype(np.uint8) * 255)


