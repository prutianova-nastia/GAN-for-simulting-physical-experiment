import tensorflow as tf
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


def sample_correlation_coefficient(X, Y):
    """
    X - sample x1, x2, .. xn
    Y - sample y1, y2, .. yn
    X, Y - np.array
    https://en.wikipedia.org/wiki/Correlation_and_dependence
    """
    x_mean, y_mean = sample_mean(X), sample_mean(Y)
    return np.sum((X - x_mean) * (Y - x_mean)) / ((X.shape[0] - 1) * unbiased_sample_variance(X) * unbiased_sample_variance(Y))

def bootstrap(X):
    rand = tf.random.uniform(shape=X.shape, dtype='int32', minval=0, maxval=X.shape[0])
    return tf.gather(X, rand)