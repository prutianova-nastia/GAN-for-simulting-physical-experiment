import tensorflow as tf
from tensorflow.keras import layers

def make_discriminator():
  return tf.keras.Sequential([
  layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same', input_shape=(8, 8, 1)),
  layers.LeakyReLU(),
  layers.Dropout(0.1),
  layers.Conv2D(32, (5, 5), strides=(2, 2), padding='same'),
  layers.LeakyReLU(),
  layers.Dropout(0.1),
  layers.Flatten(),
  layers.Dense(1),
])
