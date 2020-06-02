import os
import tensorflow as tf

GPU_NUMBER = '3'
WORK_DIR = '/home/aprutyanova/dir'


def configure_gpu():
    # Choose one of GPUs
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = GPU_NUMBER

    # Limit memory consumption
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)


def save_generator(params, generator):
    if params.GPU:
        generator.save(os.path.join(WORK_DIR, 'generator_3.h5'))
    else:
        generator.save('generator_3.h5')


def save_discriminator(params, generator):
    if params.GPU:
        generator.save(os.path.join(WORK_DIR, 'discriminator_3.h5'))
    else:
        generator.save('discriminator_3.h5')

