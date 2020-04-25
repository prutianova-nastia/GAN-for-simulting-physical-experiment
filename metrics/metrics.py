import numpy as np

from metrics.statistic_utils import standard_error, sample_mean, sample_variance

def get_max_amplitude(img):
    return np.amax(img)

def get_mean_amplitude(img):
    return np.mean(img)

def get_center(img):
    img = img.reshape((8, 8))
    y_range = np.array(list(map(lambda x: range(0, 8), range(0, 8))))
    x_range = np.transpose(y_range)
    y_mean = sample_mean(y_range, img)
    x_mean = sample_mean(x_range, img)
    return [x_mean, y_mean]

def get_covariance(img):
    img = img.reshape((8, 8))
    y_range = np.array(list(map(lambda x: range(0, 8), range(0, 8))))
    x_range = np.transpose(y_range)
    mu0 = sample_variance(x_range, img)
    mu1 = sample_variance(y_range, img)
    [x_mean, y_mean] = get_center(img)
    cov = sample_variance((x_range - x_mean) * (y_range - y_mean), img)
    return [mu0, mu1, cov]