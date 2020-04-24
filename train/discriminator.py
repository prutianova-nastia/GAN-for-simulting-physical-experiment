import tensorflow as tf

class Discriminator:
    def __init__(self, params):
        self.params = params
        self.model = params.make_discriminator()

    @tf.function
    def classify_images(self, image_batch):
        return self.model(image_batch)

    @tf.function
    def averaged_samples(self, real_images, fake_images):
        weights = tf.random.uniform(shape=(self.params.BATCH_SIZE, 1, 1, 1), dtype=tf.float32)
        return (weights * real_images) + ((1 - weights) * fake_images)

    @tf.function
    def gradient_penalty_loss(self, real_images, fake_images):
        averaged_samples = self.averaged_samples(real_images, fake_images)
        with tf.GradientTape() as t:
            t.watch(averaged_samples)
            averaged_samples_prediction = self.classify_images(averaged_samples)
        gradients = t.gradient(averaged_samples_prediction, averaged_samples)
        l2_norm = tf.sqrt(tf.reduce_sum(gradients ** 2, axis=[1, 2]))
        regularizer = tf.reduce_mean((l2_norm - 1.0) ** 2)
        return regularizer * self.params.GRADIENT_PENALTY_WEIGHT

    @tf.function
    def wasserstein_loss(self, y_true, y_pred):
        return tf.math.reduce_mean(y_true * y_pred)

    def train_on_batch(self, generator, image_batch):
        real_lables = tf.ones(shape=(self.params.BATCH_SIZE, 1), dtype=tf.float32)
        fake_lables = -tf.ones(shape=(self.params.BATCH_SIZE, 1), dtype=tf.float32)

        with tf.GradientTape() as t:
            fake_images = generator.generate_images(self.params.BATCH_SIZE)
            real_images = tf.cast(image_batch, tf.float32)
            gradient_penalty_loss = self.gradient_penalty_loss(real_images, fake_images)

            discriminator_prediction = self.classify_images(tf.concat([fake_images, real_images], 0))
            wasserstein_loss = self.wasserstein_loss(tf.concat([fake_lables, real_lables], 0), discriminator_prediction)
            discriminator_loss = gradient_penalty_loss - wasserstein_loss

        gradient = t.gradient(discriminator_loss, self.model.trainable_variables)
        self.params.discriminator_optimizer.apply_gradients(zip(gradient, self.model.trainable_variables))

        return discriminator_loss