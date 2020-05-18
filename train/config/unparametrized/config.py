import numpy as np

from tensorflow.keras.optimizers import RMSprop

from train.models.generator.generator import make_generator
from train.models.discriminator.discriminator import make_discriminator

class Params:
    def __init__(self):
        self.GPU = False
        self.parametrized = True
        self.BATCH_SIZE = 128
        self.EPOCHS = 60
        self.DISCRIMINATOR_STEPS = 1
        self.GRADIENT_PENALTY_WEIGHT = 10
        self.LATENT_DIM = 32
        self.discriminator_optimizer = RMSprop(0.0003)
        self.generator_optimizer = RMSprop(0.0003)
        self.make_generator = make_generator
        self.make_discriminator = make_discriminator
        self.min_angle = None
        self.max_angle = None

    def decrease_learning_rate(self):
        self.generator_optimizer.lr.assign(self.generator_optimizer.lr * 0.995)
        self.discriminator_optimizer.lr.assign(self.discriminator_optimizer.lr * 0.995)

    def set_angles(self, angles):
        if angles is not None:
            self.min_angle = np.amin(angles)
            self.max_angle = np.amax(angles)
