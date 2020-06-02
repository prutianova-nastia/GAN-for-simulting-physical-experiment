import tensorflow as tf

def make_discriminator():
    image_input = tf.keras.Input(shape=(8, 8, 1))
    angle_input = tf.keras.Input(shape=(1,))

    image_features_model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(filters=8, kernel_size=3, padding='same', activation='relu', input_shape=(8, 8, 1)),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Conv2D(filters=16, kernel_size=3, padding='valid', activation='relu'),
        tf.keras.layers.Dropout(0.2),

        tf.keras.layers.MaxPool2D(),

        tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding='same', activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding='same', activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding='valid', activation='relu'),
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