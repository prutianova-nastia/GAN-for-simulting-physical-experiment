import tensorflow as tf


def make_generator(LATENT_DIM):
    return tf.keras.Sequential([
        tf.keras.layers.Dense(units=216, activation='relu', input_shape=(LATENT_DIM,)),

        tf.keras.layers.Reshape((6, 6, 6)),

        tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding='same', activation='relu'),
        tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding='valid', activation='relu'),
        tf.keras.layers.UpSampling2D(),

        tf.keras.layers.Conv2D(filters=16, kernel_size=3, padding='same' , activation='relu'),
        tf.keras.layers.Conv2D(filters=16, kernel_size=3, padding='valid', activation='relu'),
        tf.keras.layers.UpSampling2D(),

        tf.keras.layers.Conv2D(filters=8, kernel_size=3, padding='valid', activation='relu'),
        tf.keras.layers.Conv2D(filters=1, kernel_size=3, padding='valid', activation='relu'),
    ])
