#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 19 13:18:52 2017

@author: jabong
"""

import numpy as np

def function(p):
    r = np.array([10*(p[1]-p[0]**2), (1-p[0])])
    fp = np.dot(np.transpose(r), r)
    J = np.array([[-20*p[0], 10], [-1, 0]])
    grad = np.dot(J.T, r.T)
    return fp, r, grad, J

def lm(p0, tol=10**(-5), maxits=100):
    nvars = np.shape(p0)[0]
    nu = 0.01
    p = p0
    fp, r, grad, J = function(p)
    e = np.sum(np.dot(np.transpose(r), r))
    nits = 0
    while nits<maxits and np.linalg.norm(grad)>tol:
        nits += 1
        fp, r, grad, J = function(p)
        H = np.dot(np.transpose(J), J) + nu*np.eye(nvars)
        
        pnew = np.zeros(np.shape(p))
        nits2 = 0
        while (p!=pnew).all() and nits2<maxits:
            nits2 += 1
            dp, resid, rank, s = np.linalg.lstsq(H, grad)
            pnew = p - dp
            fpew, rnew, gradnew, Jnew = function(pnew)
            enew = np.sum(np.dot(np.transpose(rnew), rnew))
            rho = np.linalg.norm(np.dot(np.transpose(r),r)-np.dot(np.transpose(rnew),rnew))
            rho /= np.linalg.norm(np.dot(np.transpose(grad),pnew-p))

            if rho>0:
                #update = 1
                p = pnew
                e = enew
                if rho>0.25:
                    nu /= 10
            else:
                nu *= 10
                #update = 0
        print fp, p, e, np.linalg.norm(grad), nu

p0 = np.array([-1.92,2])
lm(p0)
