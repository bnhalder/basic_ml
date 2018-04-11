#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 21 12:50:33 2017

@author: jabong
"""

from matplotlib import pyplot as plt
import numpy as np

#load iris data from sklearn
from sklearn.datasets import load_iris
data = load_iris()
features = data.data
feature_names = data.feature_names
target = data.target
target_names = data.target_names

fig,axes = plt.subplots(2,3)
pairs = [(0,1), (0,2), (0,3), (1,2), (1,3), (2,3)]

#pairs of (color,marker)
color_markers = [('r', '>'), ('g', 'o'), ('b', 'x')]

for i, (p0,p1) in enumerate(pairs):
    ax = axes.flat[i]
    for t in range(3):
        c,marker = color_markers[t]
        ax.scatter(features[target==t,p0], features[target==t,p1], marker=marker, c=c)
    ax.set_xlabel(feature_names[p0])
    ax.set_ylabel(feature_names[p1])
    ax.set_xticks([])
    ax.set_yticks([])
fig.tight_layout()
fig.savefig('figure1.png')

labels = target_names[target]

#petal length is the feature at position 2
plength = features[:, 2]
is_setosa = (labels=='setosa')
max_setosa = plength[is_setosa].max()
min_not_setosa = plength[~is_setosa].min()
print('maximum of setosa: {0}.'.format(max_setosa))
print('minimum of others: {0}.'.format(min_not_setosa))

def fit_model(features, labels):
    '''Learn a simple threshold model'''
    best_acc = -1.0
    for fi in range(features.shape[1]):
        thresh = features[:, fi].copy()
        thresh.sort()
        for t in thresh:
            pred = (features[:, fi] > t)
            acc = (pred == labels).mean()
            rev_acc = (pred == ~labels).mean()
            if rev_acc > acc:
                reverse = True
                acc = rev_acc
            else:
                reverse = False
                    
            if acc > best_acc:
                best_acc = acc
                best_fi = fi
                best_t = t
                best_reverse = reverse
    return best_t, best_fi, best_reverse

def predict(model, features):
    '''Apply a learned Model'''
    t, fi, reverse = model
    if reverse:
        return features[:, fi] <= t
    else:
        return features[:, fi] > t

def accuracy(features, labels, model):
    '''Compute the accuracy of the model'''
    preds = predict(model, features)
    return np.mean(preds==labels)

#remove setosa models as they are too easy to classify
features = features[~is_setosa]
labels = labels[~is_setosa]
is_virginica = (labels=='virginica')

#slpit the data into 2 - testing and training
testing = np.tile([True, False], 50)
training = ~testing
model = fit_model(features[training], is_virginica[training])
train_accuracy = accuracy(features[training], is_virginica[training], model)
test_accuracy = accuracy(features[testing], is_virginica[testing], model)

print('''\
      Training accuracy was {0:.1%}.
      Testing accuracy was {1:.1%} (N={2}).
      '''.format(train_accuracy, test_accuracy, testing.sum()))

COLOR_FIGURE = True
if COLOR_FIGURE:
    area1c = (1., .8, .8)
    area2c = (.8, .8, 1.)
else:
    area1c = (1., 1, 1)
    area2c = (.7, .7, .7)

t1 = 1.65
t2 = 1.75
f0, f1 = 3, 2

x0 = features[:, f0].min() * .9
x1 = features[:, f0].max() * 1.1

y0 = features[:, f1].min() * .9
y1 = features[:, f1].max() * 1.1

fig, ax = plt.subplots()
ax.fill_between([t1, x1], [y0, y0], [y1, y1], color = area2c)
ax.fill_between([x0, t1], [y0, y0], [y1, y1], color = area1c)
ax.plot([t1, t1], [y0, y1], 'k--', lw=2)
ax.plot([t2, t2], [y0, y1], 'k:', lw=2)
ax.scatter(features[is_virginica, f0], features[is_virginica, f1], c='b', marker='o', s=40)
ax.scatter(features[~is_virginica, f0], features[~is_virginica, f1], c='r', marker='x', s=40)
ax.set_ylim(y0, y1)
ax.set_xlim(x0, x1)
ax.set_xlabel(feature_names[f0])
ax.set_ylabel(feature_names[f1])
fig.tight_layout()
fig.savefig('figure2.png')




    

