import os
import tensorflow as tf

GPU_NUMBER = '1'

def configure_gpu():
    # Choose one of GPUs
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = GPU_NUMBER

    # Limit memory consumption
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)