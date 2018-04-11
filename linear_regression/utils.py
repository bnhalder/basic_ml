# This code is supporting material for the book
# Building Machine Learning Systems with Python
# by Willi Richert and Luis Pedro Coelho
# published by PACKT Publishing
#
# It is made available under the MIT License

import os
#print __file__
#print os.path.realpath(__file__)
#print os.path.dirname(__file__)

DATA_DIR = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), "data")
#print DATA_DIR

CHART_DIR = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), "charts")
#print CHART_DIR

for d in [DATA_DIR, CHART_DIR]:
    if not os.path.exists(d):
        os.mkdir(d)

