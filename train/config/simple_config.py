from tensorflow.keras.optimizers import RMSprop
from train.models.generator.simple import make_generator
from train.models.discriminator.simple import make_discriminator

class Params:
    def __init__(self):
        self.BATCH_SIZE = 64
        self.EPOCHS = 200
        self.DISCRIMINATOR_STEPS = 1
        self.GRADIENT_PENALTY_WEIGHT = 10
        self.LATENT_DIM = 32
        self.discriminator_optimizer = RMSprop(0.0001)
        self.generator_optimizer = RMSprop(0.0001)
        self.make_generator = make_generator
        self.make_discriminator = make_discriminator

    def decrease_learning_rate(self):
        self.generator_optimizer.lr.assign(self.generator_optimizer.lr * 0.994)
        self.discriminator_optimizer.lr.assign(self.discriminator_optimizer.lr * 0.994)