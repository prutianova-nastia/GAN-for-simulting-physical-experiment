import numpy as np

from read_data import normalize_data, data_to_images, read_data
from plot import plot_random_images, plot_imgs

images = data_to_images(np.array(normalize_data(read_data())))
plot_imgs(images, 4)