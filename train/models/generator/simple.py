import tensorflow as tf
from tensorflow.keras import layers

def make_generator(LATENT_DIM) :
  return tf.keras.Sequential([
  layers.Dense(8 * 8 * 32, use_bias=False, input_shape=(LATENT_DIM,), activation='relu'),
  layers.Reshape((4, 4, 128)),
  layers.Conv2DTranspose(32, (3, 3), strides=(1, 1), padding='same', activation='relu'),
  layers.Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same', activation='relu'),
  layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='relu'),
  layers.MaxPooling2D(2, 2),
  layers.Reshape((8, 8, 1)),
])
