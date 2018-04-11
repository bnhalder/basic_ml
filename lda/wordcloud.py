#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 23 10:16:58 2017

@author: jabong
"""
from __future__ import print_function

WARNED_OF_ERROR = False

def create_cloud(oname, words, maxsize=120, fontname='Lobster'):
    try:
        from pytagcloud import create_tag_image, make_tags
    except ImportError:
        if not WARNED_OF_ERROR:
            print("Could not import pytagcloud. Skipping cloud generation")
        return
    
    words = [(w, int(v*1000)) for w,v in words]
    tags = make_tags(words, maxsize=maxsize)
    create_tag_image(tags, oname, size=(1800, 1200), fontname=fontname)
    

