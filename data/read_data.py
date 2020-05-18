import numpy as np
import pandas as pd

from scipy.sparse import csr_matrix

def normalize_data(dataFrame):
    dataFrame['ipad'] -= dataFrame.min()['ipad']
    dataFrame['itime'] -= dataFrame.min()['itime']
    return dataFrame

def convert_to_images(data, parameterized=False):
    """ data - np.array """
    data = np.array([data[data[:, 0] == k] for k in np.unique(data[:, 0])])
    imgs, angles = [], []

    for i in range(0, data.shape[0]):
        row = np.array(data[i][:, 1])
        col = np.array(data[i][:, 2])
        amp = np.log(np.array(data[i][:, 3]))
        imgs.append(csr_matrix((amp, (row, col)), shape=(8, 8)).toarray())
        if parameterized:
            angles.append(data[i][:, 4][0])

    angles = np.array(angles) if parameterized else None
    return np.array(imgs), angles

def read_data(parameterized=False):
    if parameterized:
        data_frame = pd.read_csv('data/parameterized/digits.csv')
    else:
        data_frame = pd.read_csv('data/unparameterized/digits.csv')
    print((data_frame.describe()))
    return data_frame

def get_data(parameterized=False):
    images, angles = convert_to_images(np.array(normalize_data(read_data(parameterized))), parameterized)
    return np.expand_dims(images, axis=3), np.expand_dims(angles, axis=1)