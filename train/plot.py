import matplotlib.pyplot as plt

def plot_results(images, generator_loss, discriminator_loss):
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 3, 1)
    plt.imshow(images.reshape((4, 4, 8, 8)).transpose(0, 2, 1, 3).reshape(4 * 8, 4 * 8))
    plt.subplot(1, 3, 2)
    plt.plot(discriminator_loss, color='green')
    plt.xlabel('discriminator')
    plt.subplot(1, 3, 3)
    plt.plot(generator_loss, color='red')
    plt.xlabel('generator')
    plt.show()