import numpy as np
import tensorflow as tf

from train.config.config import Params
from train.generator import Generator
from train.discriminator import Discriminator
from train.plot import plot_results

def train(X_train):
    params = Params()
    generator = Generator(params)
    discriminator = Discriminator(params)

    discriminator_loss, generator_loss = [], []
    for epoch in range(params.EPOCHS):
        np.random.shuffle(X_train)
        print("Epoch: ", epoch)
        print("Number of batches: ", int(X_train.shape[0] // params.BATCH_SIZE))

        minibatches_size = params.BATCH_SIZE * params.DISCRIMINATOR_STEPS
        for i in range(int(X_train.shape[0] // (params.BATCH_SIZE * params.DISCRIMINATOR_STEPS))):
            discriminator_minibatches = X_train[i * minibatches_size: (i + 1) * minibatches_size]

            for j in range(params.DISCRIMINATOR_STEPS):
                image_batch = discriminator_minibatches[j * params.BATCH_SIZE: (j + 1) * params.BATCH_SIZE]
                discriminator_loss.append(discriminator.train_on_batch(generator, image_batch))

            generator_loss.append(generator.train_on_batch(discriminator))

        params.decrease_learning_rate()

        if epoch % 3 == 0 or epoch == 1 or epoch == 2:
            images = generator.model(tf.random.normal(shape=(16, params.LATENT_DIM)))
            plot_results(images.numpy(), generator_loss, discriminator_loss)

    return generator.model, discriminator.model