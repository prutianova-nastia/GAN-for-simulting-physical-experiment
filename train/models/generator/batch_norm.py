import tensorflow as tf

def make_generator(LATENT_DIM):
    return tf.keras.Sequential([
        tf.keras.layers.Dense(units=36, input_shape=(LATENT_DIM,)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.LeakyReLU(),

        tf.keras.layers.Reshape((6, 6, 1)),

        tf.keras.layers.Conv2DTranspose(64, 3),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.LeakyReLU(),

        tf.keras.layers.Conv2DTranspose(32, 3),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.LeakyReLU(),

        tf.keras.layers.Conv2DTranspose(32, 3, padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.LeakyReLU(),

        tf.keras.layers.Conv2DTranspose(16, 3),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.LeakyReLU(),

        tf.keras.layers.MaxPool2D(),

        tf.keras.layers.Conv2DTranspose(1, 3, activation='relu'),
    ])
