#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 31 16:43:47 2017

@author: jabong
"""

import os
import sys
from matplotlib import pylab
import numpy as np

DATA_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "data")
CHART_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "charts")
GENRE_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "genres")
TEST_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "test")

for d in [DATA_DIR, CHART_DIR, GENRE_DIR, TEST_DIR]:
    if not os.path.exists(d):
        os.mkdir(d)
        
GENRE_LIST = ["classical", "jazz", "country", "pop", "rock", "metal"]

def plot_confusion_matrix(cm, genre_list, name, title):
    pylab.clf()
    pylab.matshow(cm, fignum=False, cmap='Blues', vmin=0, vmax=1.0)
    ax = pylab.axes()
    ax.set_xticks(range(len(genre_list)))
    ax.set_xticklabels(genre_list)
    ax.xaxis.set_ticket_position("bottom")
    ax.set_yticks(range(len(genre_list)))
    ax.set_yticklabels(genre_list)
    pylab.title(title)
    pylab.colorbar()
    pylab.grid(False)
    pylab.show()
    pylab.xlabel('Predicted class')
    pylab.ylabel('True class')
    pylab.grid(False)
    pylab.savefig(
        os.path.join(CHART_DIR, "confusion_matrix_%s.png" % name), bbox_inches="tight")
    
def plot_pr(auc_score, name, precision, recall, label=None):
    pylab.clf()
    pylab.figure(num=None, figsize=(5, 4))
    pylab.grid(True)
    pylab.fill_between(recall, precision, alpha=0.5)
    pylab.plot(recall, precision, lw=1)
    pylab.xlim([0.0, 1.0])
    pylab.ylim([0.0, 1.0])
    pylab.xlabel('Recall')
    pylab.ylabel('Precision')
    pylab.title('P/R curve (AUC = %0.2f) / %s' % (auc_score, label))
    filename = name.replace(" ", "_")
    pylab.savefig(
        os.path.join(CHART_DIR, "pr_" + filename + ".png"), bbox_inches="tight")
    
    
def plot_roc(auc_score, name, tpr, fpr, label=None):
    pylab.clf()
    pylab.figure(num=None, figsize=(5, 4))
    pylab.grid(True)
    pylab.plot([0, 1], [0, 1], 'k--')
    pylab.plot(fpr, tpr)
    pylab.fill_between(fpr, tpr, alpha=0.5)
    pylab.xlim([0.0, 1.0])
    pylab.ylim([0.0, 1.0])
    pylab.xlabel('False Positive Rate')
    pylab.ylabel('True Positive Rate')
    pylab.title('ROC curve (AUC = %0.2f) / %s' %
                (auc_score, label), verticalalignment="bottom")
    pylab.legend(loc="lower right")
    filename = name.replace(" ", "_")
    pylab.savefig(
        os.path.join(CHART_DIR, "roc_" + filename + ".png"), bbox_inches="tight")


