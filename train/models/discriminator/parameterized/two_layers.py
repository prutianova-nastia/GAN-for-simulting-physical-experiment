import tensorflow as tf
from tensorflow.keras import layers


def make_discriminator():
    image_input = tf.keras.Input(shape=(8, 8, 1))
    angle_input = tf.keras.Input(shape=(1,))

    image_features_model = tf.keras.Sequential([
        layers.Conv2D(16, 3, padding='same', activation='relu', input_shape=(8, 8, 2)),
        layers.Dropout(0.2),

        layers.MaxPool2D(padding='same'),

        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.Dropout(0.2),

        layers.MaxPool2D(padding='same'),

        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.Dropout(0.2),

        layers.Flatten(),
    ])

    concatenate = tf.keras.layers.Concatenate()([angle_input, image_features_model(image_input)])

    classify = tf.keras.Sequential([
        tf.keras.layers.Dense(units=128, activation='relu'),

        layers.Dropout(0.2),

        tf.keras.layers.Dense(units=1, activation=None),
    ])

    discriminator = tf.keras.Model(
        inputs=[image_input, angle_input],
        outputs=classify(concatenate),
    )

    return discriminator
