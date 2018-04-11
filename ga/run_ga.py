#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 19 20:32:00 2017

@author: jabong
"""

import ga
import pylab as pl
import fourpeaks as fF

pl.ion()
pl.show()

plotfig = pl.figure()

ga = ga.ga(30, 'fF.fourpeaks', 301, 500, -1, 'un', 4, False)
ga.run_ga(plotfig)

pl.pause(0)