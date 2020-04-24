import numpy as np

from data.read_data import normalize_data, data_to_images, read_data
from train.train import train

def main():
    images = data_to_images(np.array(normalize_data(read_data())))
    images = np.expand_dims(images, axis=3)
    generator, discriminator = train(images)
    generator.save('saved_models/generator_.h5')
    discriminator.save('saved_models/discriminator_.h5')

if __name__ == "__main__":
    main()