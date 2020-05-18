import tensorflow as tf

class Discriminator:
    def __init__(self, params):
        self.params = params
        self.losses = []
        self.wasserstein_losses = []
        self.gradient_penalty_losses = []
        self.model = params.make_discriminator()

    def classify_images(self, image_batch, angles=None):
        model_input = self.transform_input(image_batch, angles)
        return self.model(model_input)

    def averaged_samples(self, real_images, fake_images):
        weights = tf.random.uniform(shape=(self.params.BATCH_SIZE, 1, 1, 1), dtype=tf.float32)
        return (weights * real_images) + ((1 - weights) * fake_images)

    def gradient_penalty_loss(self, real_images, fake_images, angles):
        averaged_samples = self.averaged_samples(real_images, fake_images)
        with tf.GradientTape() as t:
            t.watch(averaged_samples)
            averaged_samples_prediction = self.classify_images(averaged_samples, angles)
        gradients = t.gradient(averaged_samples_prediction, averaged_samples)
        l2_norm = tf.sqrt(tf.reduce_sum(gradients ** 2, axis=[1, 2]))
        regularizer = tf.reduce_mean((l2_norm - 1.0) ** 2)
        return regularizer * self.params.GRADIENT_PENALTY_WEIGHT

    def wasserstein_loss(self, y_true, y_pred):
        return tf.math.reduce_mean(y_true * y_pred)

    def transform_input(self, image_batch, angles=None):
        if self.params.parametrized:
            assert angles is not None
            flatten = tf.reshape(image_batch, shape=(image_batch.shape[0], image_batch.shape[1] * image_batch.shape[2],))
            # angles = tf.reshape(tf.cast(angles, tf.float32), shape=(angles.shape[0], 1))
            return tf.concat([flatten, angles], axis=1)
        else:
            return image_batch

    def train_on_batch(self, generator, image_batch, real_angles=None):
        real_lables = tf.ones(shape=(self.params.BATCH_SIZE, 1), dtype=tf.float32)
        fake_lables = -tf.ones(shape=(self.params.BATCH_SIZE, 1), dtype=tf.float32)

        with tf.GradientTape() as t:
            fake_images, fake_angles = generator.generate_images(self.params.BATCH_SIZE, real_angles)
            real_images = tf.cast(image_batch, tf.float32)
            gradient_penalty_loss = self.gradient_penalty_loss(real_images, fake_images, real_angles)
            self.gradient_penalty_losses.append(gradient_penalty_loss)

            fake_images, fake_angles = generator.generate_images(self.params.BATCH_SIZE)
            angles = tf.concat([fake_angles, real_angles], axis=0) if self.params.parametrized else None
            discriminator_prediction = self.classify_images(tf.concat([fake_images, real_images], axis=0), angles)
            wasserstein_loss = self.wasserstein_loss(tf.concat([fake_lables, real_lables], axis=0), discriminator_prediction)
            self.wasserstein_losses.append(wasserstein_loss)

            discriminator_loss = gradient_penalty_loss - wasserstein_loss

        gradient = t.gradient(discriminator_loss, self.model.trainable_variables)
        self.params.discriminator_optimizer.apply_gradients(zip(gradient, self.model.trainable_variables))
        self.losses.append(discriminator_loss)
        return discriminator_loss