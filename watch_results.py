import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras.models import load_model

from data.read_data import normalize_data, data_to_images, read_data
from train.config.config import Params
from metrics.plot import plot_results

N = 4

def draw_imgs(generated, original):
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(generated.reshape((N, N, 8, 8)).transpose(0, 2, 1, 3).reshape(N * 8, N * 8))
    plt.subplot(1, 2, 2)
    plt.imshow(original.reshape((N, N, 8, 8)).transpose(0, 2, 1, 3).reshape(N * 8, N * 8))
    plt.show()

images = data_to_images(np.array(normalize_data(read_data())))
images = np.expand_dims(images, axis=3)
generator = load_model('saved_models/generator_500.h5')
generated_images = generator(tf.random.normal(shape=(N * N, Params().LATENT_DIM)))
# draw_imgs(images[:N * N], generated_images.numpy())

real = images[:3000]
generated = generator(tf.random.normal(shape=(3000, Params().LATENT_DIM))).numpy()

plot_results(real, generated)


