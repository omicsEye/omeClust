import math

from numpy import array
from scipy.stats import spearmanr

from . import config


def pDistance(x, y):
    pMetric = c_hash_metric[config.similarity_method]
    dist = math.fabs(1.0 - math.fabs(pMetric(x, y)))
    return dist


def spearman(x, y):
    x = array(x)
    y = array(y)
    if x.ndim > 1:
        x = x[0]
    if y.ndim > 1:
        y = y[0]
    return spearmanr(x, y, nan_policy='omit')[0]


c_hash_metric = {
    "spearman": spearman
}
