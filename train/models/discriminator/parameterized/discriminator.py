import tensorflow as tf
from tensorflow.keras import layers


def make_discriminator():
    return tf.keras.Sequential([
        layers.Dense(256, input_shape=(65,)),

        layers.Reshape((16, 16, 1)),

        layers.Conv2D(16, 3, padding='same', activation='relu', input_shape=(8, 8, 1)),
        layers.Dropout(0.2),

        layers.MaxPool2D(padding='same'),

        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.Dropout(0.2),

        layers.MaxPool2D(padding='same'),

        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.Dropout(0.2),

        layers.Flatten(),

        layers.Dense(128, activation='relu'),
        layers.Dropout(0.4),

        layers.Dense(1),
    ])