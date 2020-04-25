import numpy as np
import math

def sample_mean(X, weights=None):
    """ X - sample x1, x2, .. xn
    X - np.array
    returns sample mean
    """
    if weights is None:
        weights = np.ones(X.shape)
    return np.sum(X * weights) / np.sum(weights)

def sample_variance(X, weights=None):
    """ X - sample x1, x2, .. xn
    X - np.array
    returns sample varience ** 2
    """
    mean = sample_mean(X, weights)
    if weights is None:
        weights = np.ones(X.shape)
    sqr_variance = np.sum(weights * ((X - mean) ** 2)) / np.sum(weights)
    return math.sqrt(sqr_variance)

def unbiased_sample_variance(X, weights=None):
    """
    https://en.wikipedia.org/wiki/Unbiased_estimation_of_standard_deviation
    """
    if weights is None:
        weights = np.ones(X.shape)
    n = np.sum(weights)
    return sample_variance(X, weights) * math.sqrt(n) / math.sqrt(n - 1)

def standard_error(X):
    return math.sqrt(unbiased_sample_variance(X)) / math.sqrt(X.shape[0])


# rrange = np.array(list(map(lambda x: range(0, 3), range(0, 3))))
# print(rrange)
# w = np.array([[10, 20, 0], [30, 40, 1], [10, 10, 10]])
# print(sample_mean(rrange, w))
