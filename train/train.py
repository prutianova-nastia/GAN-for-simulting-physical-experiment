import tensorflow as tf

from train.generator import Generator
from train.discriminator import Discriminator
from train.plot import plot_results
from GPU.utils import save_generator

def random_shuffle(images, angles):
    if angles is None:
        return tf.random.shuffle(images), angles
    indices = tf.range(start=0, limit=tf.shape(images)[0], dtype=tf.int32)
    shuffled_indices = tf.random.shuffle(indices)
    return tf.gather(images, shuffled_indices), tf.gather(angles, shuffled_indices)


def train(params, images, angles=None):
    generator = Generator(params)
    discriminator = Discriminator(params)

    for epoch in range(params.EPOCHS):
        images, angles = random_shuffle(images, angles)
        angles = tf.reshape(tf.cast(angles, dtype=tf.float32), shape=(angles.shape[0], 1))

        print("Epoch: ", epoch)
        print("Number of batches: ", int(images.shape[0] // params.BATCH_SIZE))

        minibatches_size = params.BATCH_SIZE * params.DISCRIMINATOR_STEPS

        for i in range(int(images.shape[0] // (params.BATCH_SIZE * params.DISCRIMINATOR_STEPS))):
            image_minibatches = images[i * minibatches_size: (i + 1) * minibatches_size]
            if params.parametrized:
                angle_minibatches = angles[i * minibatches_size: (i + 1) * minibatches_size]

            for j in range(params.DISCRIMINATOR_STEPS):
                image_batch = image_minibatches[j * params.BATCH_SIZE: (j + 1) * params.BATCH_SIZE]
                if params.parametrized:
                    angle_batch = angle_minibatches[j * params.BATCH_SIZE: (j + 1) * params.BATCH_SIZE]
                else:
                    angle_batch = None

                discriminator.train_on_batch(generator, image_batch, angle_batch)

            generator.train_on_batch(discriminator)
        params.decrease_learning_rate()
        save_generator(params, generator.model)

        imgs, _ = generator.generate_images(16)

        plot_results(imgs.numpy(), generator.losses, discriminator.losses)

    return generator.model, discriminator.model
