from data.read_data import get_data
from train.train import train
from GPU.utils import configure_gpu, save_generator
from train.config.parameterized.config import Params

def main():
    params = Params()
    if params.GPU:
        configure_gpu()
    images, angles = get_data(params.parametrized)
    params.set_angles(angles)
    generator, discriminator = train(params, images, angles)
    save_generator(params, generator)

if __name__ == "__main__":
    main()