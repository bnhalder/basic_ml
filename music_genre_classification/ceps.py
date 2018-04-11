#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 31 17:39:02 2017

@author: jabong
"""

import os
import glob
import sys

import numpy as np
import scipy
import scipy.io.wavfile
from scikits.talkbox.features import mfcc
from utils import GENRE_DIR

def write_ceps(ceps, fn):
    base_fn, ext = os.path.splitext(fn)
    data_fn = base_fn + ".ceps"
    np.save(data_fn, ceps)
    print("Written %s"%data_fn)
    
def create_ceps(fn):
    sample_rate, X = scipy.io.wavfile.read(fn)
    ceps, mspec, spec = mfcc(X)
    write_ceps(ceps, fn)

def read_ceps(genre_list, base_dir=GENRE_DIR):
    X = []
    y = []
    for label, genre in enumerate(genre_list):
        for fn in glob.glob(os.path.join(base_dir, genre, "*.ceps.npy")):
            ceps = np.load(fn)
            num_ceps = len(ceps)
            X.append(np.mean(ceps[int(num_ceps * 0.1) : int(num_ceps * 0.9)], axis=0))
            y.append(label)
    return np.array(X), np.array(y)

if __name__ == "__main__":
    #os.chdir(GENRE_DIR)
    #glob_wav = os.path.join(sys.argv[1], "*.wav")
    #print(glob_wav)
    #for fn in glob.glob(glob_wav):
    #    create_ceps(fn)
    genres = ["classical", "jazz", "country", "pop", "rock", "metal"]
    for genre in genres:
        for fn in glob.glob(os.path.join(GENRE_DIR, genre, "*.wav")):
            create_ceps(fn)
    