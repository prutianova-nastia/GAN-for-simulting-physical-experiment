import numpy as np
import pandas as pd

from scipy.sparse import csr_matrix

def normalize_data(dataFrame):
    dataFrame['ipad'] -= dataFrame.min()['ipad']
    dataFrame['itime'] -= dataFrame.min()['itime']
    return dataFrame

def data_to_images(data):
    # data ~ np.array
    data = np.array([data[data[:, 0] == k] for k in np.unique(data[:, 0])])
    imgs = []

    for i in range(0, data.shape[0]):
        row = np.array(data[i][:, 1])
        col = np.array(data[i][:, 2])
        amp = np.log(np.array(data[i][:, 3]))
        imgs.append(csr_matrix((amp, (row, col)), shape=(8, 8)).toarray())

    imgs = np.array(imgs)
    return imgs

def read_data():
    return pd.read_csv('data/digits.csv')

