import numpy as np
import scipy.io
import os
import matplotlib.pyplot as plt
from PIL import Image

from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import callbacks
from tensorflow.keras.callbacks import ModelCheckpoint


class AppleTree(object):
    def __init__(self, mode, path):
        self.mode = mode
        self.path = path
        self.model = load_model(path)

        # load in data to evaluate (will eventually be data from chosen image, using test for now)
        self.sample_data = np.load('data/{}/{}_balanced_data.npz'.format(self.mode,self.mode))
        self.x_test_sample = self.sample_data['arr_2']
        self.x_test_sample = self.x_test_sample.astype('float32') / 255. # data was uint8 [0-255]

    def predict(self):
        self.pred_probs = self.model.predict(self.x_test_sample)
        print('Probabilities: {}'.format(self.pred_probs))
        self.y_pred = np.argmax(self.model.predict(self.x_test_sample), axis=1)
        print('Guesses: {}'.format(self.y_pred))

if __name__ == "__main__":
    trees = AppleTree(mode='RGB',
                     path='data/appleseed.h5')
    
    trees.predict()
    

