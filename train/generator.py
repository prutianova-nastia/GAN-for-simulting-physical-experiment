import tensorflow as tf

class Generator:
    def __init__(self, params):
        self.params = params
        self.model = params.make_generator(params.LATENT_DIM)

    @tf.function
    def generate_images(self, images_number):
        return self.model(tf.random.normal(shape=(images_number, self.params.LATENT_DIM)))

    @tf.function
    def wasserstein_loss(self, y_true, y_pred):
        return tf.math.reduce_mean(y_true * y_pred)

    @tf.function
    def train_on_batch(self, discriminator):
        fake_lables = -1 * tf.ones(shape=(self.params.BATCH_SIZE, 1))

        with tf.GradientTape() as t:
            generated_images = self.generate_images(self.params.BATCH_SIZE)
            discriminator_prediction = discriminator.classify_images(generated_images)
            generator_loss = self.wasserstein_loss(fake_lables, discriminator_prediction)

        gradient = t.gradient(generator_loss, self.model.trainable_variables)
        self.params.generator_optimizer.apply_gradients(zip(gradient, self.model.trainable_variables))
        return generator_loss