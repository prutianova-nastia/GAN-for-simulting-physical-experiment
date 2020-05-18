import os
import tensorflow as tf

GPU_NUMBER = '2'
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
        generator.save(os.path.join(WORK_DIR, 'generator1.h5'))
    else:
        generator.save('generator1.h5')