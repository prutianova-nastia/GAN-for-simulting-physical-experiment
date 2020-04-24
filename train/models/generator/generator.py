import tensorflow as tf

def make_generator(LATENT_DIM):
    return tf.keras.Sequential([
    tf.keras.layers.Dense(units=64, activation='relu', input_shape=(LATENT_DIM,)),

    tf.keras.layers.Dense(units=360, activation='relu'),
    tf.keras.layers.Reshape((3, 3, 40)),

    tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding='same', activation='relu'),
    tf.keras.layers.UpSampling2D(), # 8x8

    tf.keras.layers.Conv2D(filters=16, kernel_size=3, padding='same', activation='relu'),
    tf.keras.layers.UpSampling2D(), # 16x16

    tf.keras.layers.Conv2D(filters=8, kernel_size=3, padding='same', activation='relu'),
    tf.keras.layers.Conv2D(filters=4, kernel_size=3, padding='valid', activation='relu'),

    tf.keras.layers.Conv2D(filters=1, kernel_size=3, padding='valid', activation='relu'),
])