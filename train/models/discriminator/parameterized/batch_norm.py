import tensorflow as tf

def make_discriminator():
    image_input = tf.keras.Input(shape=(8, 8, 1))
    angle_input = tf.keras.Input(shape=(1,))

    image_features_model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(16, 3, input_shape=(8, 8, 1)),
        tf.keras.layers.LeakyReLU(),
        tf.keras.layers.Dropout(0.2),

        tf.keras.layers.Conv2D(16, 3, padding='same'),
        tf.keras.layers.LeakyReLU(),
        tf.keras.layers.Dropout(0.2),

        tf.keras.layers.Conv2D(32, 3),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.LeakyReLU(),

        tf.keras.layers.Dropout(0.2),

        tf.keras.layers.Conv2D(32, 3, padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.LeakyReLU(),

        tf.keras.layers.Dropout(0.2),

        tf.keras.layers.Conv2D(64, 3),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.LeakyReLU(),

        tf.keras.layers.Dropout(0.2),

        tf.keras.layers.Flatten()
    ])

    concatenate = tf.keras.layers.Concatenate()([angle_input, image_features_model(image_input)])

    classify = tf.keras.Sequential([
        tf.keras.layers.Dense(units=128, activation='relu'),

        tf.keras.layers.Dropout(0.2),

        tf.keras.layers.Dense(units=1, activation=None),
    ])

    discriminator = tf.keras.Model(
        inputs=[image_input, angle_input],
        outputs=classify(concatenate),
    )

    return discriminator