import matplotlib.pyplot as plt
from random import *

def plot_random_images(imgs, shape):
    plt.figure(figsize=(15, 5))
    size = imgs.shape[0]
    for i in range(1, shape[0] + 1):
        for j in range(1, shape[1] + 1):
            plt.subplot(shape[0], shape[1], (i - 1) * shape[0] + j)
            plt.imshow(imgs[randint(0, size)])
    plt.show()

def plot_imgs(imgs, w):
    plt.imshow(imgs[:(w ** 2)].reshape(w, w, 8, 8).transpose(0, 2, 1, 3).reshape(w * 8, w * 8))
    plt.show()


