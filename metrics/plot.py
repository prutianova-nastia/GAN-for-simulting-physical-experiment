import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import math

from metrics.metrics import get_max_amplitude, get_mean_amplitude, get_center, get_covariance
from metrics.statistic_utils import sample_mean, standard_error

def get_centers(imgs):
    return np.array(list(map(lambda img: get_center(img), imgs)))

def get_amps(imgs):
    return np.array(list(map(lambda img: get_max_amplitude(img), imgs)))

def get_covariances(imgs):
    return np.array(list(map(lambda img: get_covariance(img), imgs)))

def plot_centers(real, generated):
    plt.scatter(real[:, 0], real[:, 1], color='limegreen')
    plt.scatter(generated[:, 0], generated[:, 1], color='gold')
    plt.show()

def distplot(real, generated):
    sns.distplot(real)
    sns.distplot(generated)
    plt.show()

def show_mistake(real_dist, generated_dist, name=""):
    print(name)
    real_mean = sample_mean(real_dist)
    generated_mean = sample_mean(generated_dist)
    print("Real mean: ", real_mean, ", Generated mean: ", generated_mean )
    print("Real standard mistake: ", standard_error(real_dist), ", Mistake: ", abs(generated_mean - real_mean))

def plot_results(real, generated):
    real_centers = get_centers(real)
    generated_centers = get_centers(generated)

    plot_centers(real_centers, generated_centers)

    show_mistake(real_centers[:, 0], generated_centers[:, 0], "X_center")
    show_mistake(real_centers[:, 1], generated_centers[:, 1], "Y_center")

    real_amps = get_amps(real)
    generated_amps = get_amps(generated)

    distplot(real_amps, generated_amps)
    show_mistake(real_amps, generated_amps, "Amplitude")

    real_cov = get_covariances(real)
    generated_cov = get_covariances(generated)

    for i in range(0, 3):
        distplot(real_cov[:, i], generated_cov[:, i])
        show_mistake(real_cov[:, i], generated_cov[:, i], "Covariance")



