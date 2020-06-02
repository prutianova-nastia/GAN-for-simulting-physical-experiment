import matplotlib.pyplot as plt

def plot_results(images, generator_loss, discriminator_loss, wasserstein, penalty, gradient_norm=None):
    plt.figure(figsize=(25, 6))
    plt.subplot(1, 5, 1)
    plt.imshow(images.reshape((4, 4, 8, 8)).transpose(0, 2, 1, 3).reshape(4 * 8, 4 * 8))
    plt.subplot(1, 5, 2)
    plt.plot(discriminator_loss, color='green')
    plt.xlabel('discriminator')
    plt.subplot(1, 5, 3)
    plt.plot(generator_loss, color='red')
    plt.xlabel('generator')
    plt.subplot(1, 5, 4)
    plt.plot(wasserstein, color='blue')
    plt.xlabel('wasserstein')
    plt.subplot(1, 5, 5)
    plt.plot(penalty, color='yellow')
    plt.xlabel('penalty')
    plt.show()