#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 25 23:41:32 2017

@author: jabong
"""

import time
start_time = time.time()

import numpy as np
from sklearn.metrics import precision_recall_curve, roc_curve, auc
from sklearn.cross_validation import ShuffleSplit

from utils import plot_pr
from utils import load_data
from utils import tweak_labels

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB

def create_ngram_model():
    tfidf_ngrams = TfidfVectorizer(ngram_range=(1,3), analyzer='word', binary=False)
    clf = MultinomialNB()
    pipeline = Pipeline([('vect', tfidf_ngrams), ('clf', clf)])
    return pipeline

def train_model(clf_factory, X, Y, name="NB_Gram", plot=False):
    cv = ShuffleSplit(n=len(X), n_iter=10, test_size=0.3, random_state=0)
    train_errors = []
    test_errors = []
    
    scores = []
    pr_scores = []
    precisions, recalls, thresholds = [], [], []
    
    for train, test in cv:
        X_train, y_train = X[train], Y[train]
        X_test, y_test = X[test], Y[test]
        
        clf = clf_factory()
        clf.fit(X_train, y_train)
        
        train_score = clf.score(X_train, y_train)
        test_score = clf.score(X_test, y_test)
        
        train_errors.append(1-train_score)
        test_errors.append(1-test_score)
        
        scores.append(test_score)
        proba = clf.predict_proba(X_test)
        
        fpr, tpr, roc_thresholds = roc_curve(y_test, proba[:, 1])
        precision, recall, pr_thresholds = precision_recall_curve(y_test, proba[:, 1])
        pr_scores.append(auc(recall, precision))
        precisions.append(precision)
        recalls.append(recall)
        thresholds.append(pr_thresholds)
    scores_to_sort = pr_scores
    median = np.argsort(scores_to_sort)[len(scores_to_sort)/2]
    
    if plot:
        plot_pr(pr_scores[median], name, "01", precisions[median], recalls[median], label=name)
        summary = (np.mean(scores), np.std(scores),
               np.mean(pr_scores), np.std(pr_scores))
        print "%.3f\t%.3f\t%.3f\t%.3f\t" % summary
    
    return np.mean(train_errors), np.mean(test_errors)

if __name__ == "__main__":
    X_orig, Y_orig = load_data()
    classes = np.unique(Y_orig)
    for c in classes:
        print "#%s: %i" % (c, sum(Y_orig == c))
    print "== Pos vs. neg =="
    pos_neg = np.logical_or(Y_orig == "Pos", Y_orig == "Neg")
    X = X_orig[pos_neg]
    Y = Y_orig[pos_neg]
    Y = tweak_labels(Y, ["Pos"])
    train_model(create_ngram_model, X, Y, name="pos vs neg", plot=True)
    
    print "== Pos/neg vs. irrelevant/neutral =="
    X = X_orig
    Y = tweak_labels(Y_orig, ["Pos", "Neg"])
    train_model(create_ngram_model, X, Y, name="sent vs rest", plot=True)
    
    print "== Pos vs. rest =="
    X = X_orig
    Y = tweak_labels(Y_orig, ["Pos"])
    train_model(create_ngram_model, X, Y, name="pos vs rest", plot=True)

    print "== Neg vs. rest =="
    X = X_orig
    Y = tweak_labels(Y_orig, ["Neg"])
    train_model(create_ngram_model, X, Y, name="neg vs rest", plot=True)

    print "time spent:", time.time() - start_time
    