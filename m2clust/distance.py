

import math
import scipy
import scipy.cluster
from scipy.spatial.distance import cdist
import scipy.stats
from scipy.stats import spearmanr
from . import config
from numpy import array

def pDistance(x, y):
    pMetric = c_hash_metric[config.similarity_method]
    dist = math.fabs(1.0 - math.fabs(pMetric(x, y)))
    return  dist

def spearman(X, Y):
    X = array(X)
    Y = array(Y)
    if X.ndim > 1: 
        X = X[0]
    if Y.ndim > 1:
        Y = Y[0]
    return spearmanr(X, Y, nan_policy='omit')[0]

c_hash_metric = {
                "spearman": spearman
                }