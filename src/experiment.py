import scipy.io
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from appleseed_cnn import AppleseedCNN

# create a locally manageable dataset
mat = scipy.io.loadmat('data/sat-6-full.mat')
apples = AppleseedCNN(mat)

x_train_play = apples.x_train_play
y_train_play = apples.y_train_play
x_test_play = apples.x_test_play
y_test_play = apples.y_test_play

np.savez('data/play_data', x_train_play, y_train_play, x_test_play, y_test_play)


