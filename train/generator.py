import tensorflow as tf

class Generator:
    def __init__(self, params, model=None):
        self.params = params
        self.losses = []
        if model is None:
            self.model = params.make_generator(params.LATENT_DIM)
        else:
            self.model = model

    def generate_angles(self, number):
        assert self.params.parametrized
        return tf.random.uniform(shape=(number, 1), minval=self.params.min_angle, maxval=self.params.max_angle,
                                 dtype=tf.float32)

    def generate_random_noise(self, shape):
        return tf.random.normal(shape=shape)

    def generate_images(self, images_number, angles=None):
        if self.params.parametrized:
            if angles is None:
                angles = self.generate_angles(images_number)
            else:
                assert angles.shape[0] == images_number
            model_input = tf.concat([angles, self.generate_random_noise((images_number, self.params.LATENT_DIM - 1))],
                                    axis=1)
        else:
            model_input = self.generate_random_noise((images_number, self.params.LATENT_DIM))
        return self.model(model_input), angles

    def wasserstein_loss(self, y_true, y_pred):
        return tf.math.reduce_mean(y_true * y_pred)

    def train_on_batch(self, discriminator):
        fake_lables = -1 * tf.ones(shape=(self.params.BATCH_SIZE, 1))

        with tf.GradientTape() as t:
            generated_images, generated_angles = self.generate_images(self.params.BATCH_SIZE, None)
            discriminator_prediction = discriminator.classify_images(generated_images, generated_angles)
            generator_loss = self.wasserstein_loss(fake_lables, discriminator_prediction)

        gradient = t.gradient(generator_loss, self.model.trainable_variables)
        self.params.generator_optimizer.apply_gradients(zip(gradient, self.model.trainable_variables))
        self.losses.append(generator_loss)
        return generator_loss