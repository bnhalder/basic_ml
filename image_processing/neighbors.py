#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  1 15:52:27 2017

@author: jabong
"""

import numpy as np
import mahotas as mh
from glob import glob
from features import texture, chist
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from scipy.spatial import distance

basedir = 'SimpleImageDataset/'


haralicks = []
chistogram = []

print('Computing features...')
# Use glob to get all the images
images = glob('{}/*.jpg'.format(basedir))
# We sort the images to ensure that they are always processed in the same order
# Otherwise, this would introduce some variation just based on the random
# ordering that the filesystem uses
images.sort()

for fname in images:
    imc = mh.imread(fname)
    imc = imc[200:-200,200:-200]
    haralicks.append(texture(mh.colors.rgb2grey(imc)))
    chistogram.append(chist(imc))

haralicks = np.array(haralicks)
chistogram = np.array(chistogram)
features = np.hstack([chistogram, haralicks])

print('Computing Neighbors...')
sc = StandardScaler()
features = sc.fit_transform(features)
dists = distance.squareform(distance.pdist(features))

print('Plotting...')
fig, axes = plt.subplots(2, 9, figsize=(16,8))

# Remove ticks from all subplots
for ax in axes.flat:
    ax.set_xticks([])
    ax.set_yticks([])

for ci,i in enumerate(range(0,90,10)):
    left = images[i]
    dists_left = dists[i]
    right = dists_left.argsort()
    # right[0] is the same as left[i], so pick the next closest element
    right = right[1]
    right = images[right]
    left = mh.imread(left)
    right = mh.imread(right)
    axes[0, ci].imshow(left)
    axes[1, ci].imshow(right)

fig.tight_layout()
fig.savefig('figure_neighbors.png', dpi=300)