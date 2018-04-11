#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 25 19:09:04 2017

@author: jabong
"""
import os
import collections
import csv
import json
from nltk.corpus import sentiwordnet

from matplotlib import pylab
import numpy as np

DATA_DIR = "data"
CHART_DIR = "charts"

if not os.path.exists(DATA_DIR):
    raise RuntimeError("Expecting directory 'data' in current path")

if not os.path.exists(CHART_DIR):
    os.mkdir(CHART_DIR)

def tweak_labels(Y, pos_sent_list):
    pos = Y == pos_sent_list[0]
    for sent_label in pos_sent_list[1:]:
        pos |= Y == sent_label
    Y = np.zeros(Y.shape[0])
    Y[pos] = 1
    Y = Y.astype(int)
    return Y

def load_data():
    tweets = []
    labels = []
    lines = []
    with open(os.path.join(DATA_DIR, "corpus2.csv"), "r") as f:
        for line in f:
            lines.append(line.strip().split(','))
    for line in lines:
        label, tweet = line
        tweets.append(tweet)
        labels.append(label)
    tweets = np.asarray(tweets)
    labels = np.asarray(labels)
    return tweets, labels

tweets, labels = load_data()

def plot_pr(auc_score, name, phase, precision, recall, label=None):
    pylab.clf()
    pylab.figure(num=None, figsize=(5, 4))
    pylab.grid(True)
    pylab.fill_between(recall, precision, alpha=0.5)
    pylab.plot(recall, precision, lw=1)
    pylab.xlim([0.0, 1.0])
    pylab.ylim([0.0, 1.0])
    pylab.xlabel('Recall')
    pylab.ylabel('Precision')
    pylab.title('P/R curve (AUC=%0.2f) / %s' % (auc_score, label))
    filename = name.replace(" ", "_")
    pylab.savefig(os.path.join(CHART_DIR, "pr_%s_%s.png" %
                  (filename, phase)), bbox_inches="tight")
    
def log_false_positives(clf, X, y, name):
    with open("FP_"+name.replace(" ", "_")+".tsv", "w") as f:
        false_positive = clf.predict(X) != y
        for tweet, false_class in zip(X[false_positive], y[false_positive]):
            f.write("%s\t%s\n" % (false_class, tweet.encode("ascii", "ignore")))

def load_sent_word_net():
    sent_scores = collections.defaultdict(list)
    with open(os.path.join(DATA_DIR, "SentiWordNet.txt"), "r") as csvfile:
        reader = csv.reader(csvfile, delimiter='\t', quotechar='"')
        for line in reader:
            if line[0].startswith('#'):
                continue
            if len(line)==1:
                continue
            
            POS, ID, PosScore, NegScore, SysnetTerms, Gloss = line
            if len(POS) == 0 or len(ID) == 0:
                continue
            for term in SysnetTerms.split(" "):
                # drop #number at the end of every term
                term = term.split("#")[0]
                term = term.replace("-", " ").replace("_", " ")
                key = "%s/%s" % (POS, term.split("#")[0])
                sent_scores[key].append((float(PosScore), float(NegScore)))
    for key, value in sent_scores.iteritems():
        sent_scores[key] = np.mean(value, axis=0)

    return sent_scores
            
    
    

    




            
            
        
        
    
    
    