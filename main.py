from data.read_data import get_data
from train.train import train
from GPU.utils import configure_gpu, save_generator
from train.config.parameterized.config import Params
from sklearn.model_selection import train_test_split

def main():
    params = Params()
    if params.GPU:
        configure_gpu()
    images, angles = get_data(params.parametrized)
    params.set_angles(angles)

    X_train, X_test, y_train, y_test = train_test_split(images, angles, test_size=0.2, random_state=42)

    generator, discriminator = train(params, X_train, y_train)
    save_generator(params, generator)

if __name__ == "__main__":
    main()