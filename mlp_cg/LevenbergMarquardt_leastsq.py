#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 19 14:50:00 2017

@author: jabong
"""

import pylab as pl
import numpy as np

def function(p, x, ydata):
    fp = p[0]*np.cos(p[1]*x)+ p[1]*np.sin([p[0]*x])
    r = ydata - fp
    J = np.transpose([-np.cos(p[0]*x)-p[1]*np.cos(p[0]*x)*x, p[0] * np.sin(p[1]*x)*x-np.sin(p[0]*x)])
    grad = np.dot(J.T,r.T)
    return fp,r,grad,J

def lm(p0, x, f, tol=10**(-5), maxits=100):
    nvars=np.shape(p0)[0]
    nu=0.01
    p = p0
    fp,r,grad,J = function(p,x,f)
    e = np.sum(np.dot(np.transpose(r),r))
    nits = 0
    
    while nits<maxits and np.linalg.norm(grad)>tol:
        nits += 1
        fp,r,grad,J = function(p,x,f)
        H=np.dot(np.transpose(J),J) + nu*np.eye(nvars)
        pnew = np.zeros(np.shape(p))
        nits2 = 0
        while (p!=pnew).all() and nits2<maxits:
            nits2 += 1
            dp,resid,rank,s = np.linalg.lstsq(H,grad)
            pnew = p - dp[:,0]
            
            fpnew,rnew,gradnew,Jnew = function(pnew,x,f)
            enew = np.sum(np.dot(np.transpose(rnew),rnew))
            rho = np.linalg.norm(np.dot(np.transpose(r),r)-np.dot(np.transpose(rnew),rnew))
            rho /= np.linalg.norm(np.dot(np.transpose(grad),pnew-p))
            
            if rho>0:
                p = pnew
                e = enew
                if rho > 0.25:
                    nu /= 10
            else:
                nu *= 10
        print p, e, np.linalg.norm(grad), nu
    return p

#p0 = np.array([100.5,102.5])
p0 = np.array([101, 101])
p = np.array([100,102])

x = np.arange(0,2*np.pi,0.1)
y = p[0]*np.cos(p[1]*x)+ p[1]*np.sin([p[0]*x]) + np.random.rand(len(x))
p = lm(p0,x,y)
y1 = p[0]*np.cos(p[1]*x)+ p[1]*np.sin([p[0]*x])

pl.plot(x,np.squeeze(y),'-')
pl.plot(x,np.squeeze(y1),'r--')
pl.legend(['Actual Data','Fitted Data'])
pl.show()
    