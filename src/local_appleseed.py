import numpy as np
import scipy.io

from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

class LocalSeeds(object):
    def __init__(self, play_data):
        # load in test and train arrays saved as npz file
        self.play_data = play_data
        
        # create a locally manageable dataset
        self.x_train_play = play_data['arr_0']
        self.y_train_play = play_data['arr_1']
        self.x_test_play = play_data['arr_2']
        self.y_test_play = play_data['arr_3']

        # .predict() doesn't like uint8
        self.x_train_play = self.x_train_play.astype('float32') / 255. # data was uint8 [0-255]
        self.x_test_play = self.x_test_play.astype('float32') / 255. # data was uint8 [0-255]

    def define_model(self, nb_filters, kernel_size, input_shape, pool_size):
        model = Sequential() # not sure what the other options are for this

        # note: the convolutional layers and dense layers require an activation function
        # see https://keras.io/activations/
        # and https://en.wikipedia.org/wiki/Activation_function
        # options: 'linear', 'sigmoid', 'tanh', 'relu', 'softplus', 'softsign'

        model.add(Conv2D(nb_filters, (kernel_size[0], kernel_size[1]),
                            padding='valid', 
                            input_shape=input_shape)) #first conv. layer  KEEP
        model.add(Activation('relu')) # Activation specification necessary for Conv2D and Dense layers

        model.add(Conv2D(nb_filters, (kernel_size[0], kernel_size[1]), padding='valid')) #2nd conv. layer KEEP
        model.add(Activation('relu'))

        model.add(MaxPooling2D(pool_size=pool_size)) # decreases size, helps prevent overfitting
        model.add(Dropout(0.5)) # zeros out some fraction of inputs, helps prevent overfitting

        model.add(Flatten()) # necessary to flatten before going into conventional dense layer  KEEP
        print('Model flattened out to ', model.output_shape)

        # start a typical neural network
        model.add(Dense(32))
        model.add(Activation('relu'))

        model.add(Dropout(0.175)) # zeros out some fraction of inputs, helps prevent overfitting

        model.add(Dense(nb_classes)) # 6 final nodes (one for each class)  KEEP
        model.add(Activation('softmax')) # softmax at end to pick between classes 0-6

        model.compile(loss='categorical_crossentropy',
                    optimizer='adam',
                    metrics=['accuracy'])
        self.model = model

    def fit_model(self):
        self.model.fit(self.x_train_play, self.y_train_play, batch_size=batch_size, epochs=nb_epoch,
        verbose=1, validation_data=(self.x_test_play, self.y_test_play))
    
    def evaluate_model(self):
        score = self.model.evaluate(self.x_test_play, self.y_test_play, verbose=0)
        print('Test score:', score[0])
        print('Test accuracy:', score[1]) # the important one

if __name__ == "__main__":

    print('Creating class')
    play_data = np.load('data/play_data.npz')
    seeds = LocalSeeds(play_data)

    print("Initializing parameters")
    batch_size = 10  # number of training samples used at a time to update the weights
    nb_classes = 6   # number of output possibilites: [0 - 9] KEEP
    nb_epoch = 10   # number of passes through the entire train dataset before weights "final"
    img_rows, img_cols = 28, 28  # the size of the NAIP images
    input_shape = (img_rows, img_cols, 4)  # 4 channel image input (RGBA) 
    nb_filters = 8 # number of convolutional filters to use
    pool_size = (2, 2) # pooling decreases image size, reduces computation, adds translational invariance
    kernel_size = (4, 4) # convolutional kernel size, slides over image to learn features

    print('Building model')
    seeds.define_model(nb_filters, kernel_size, input_shape, pool_size)

    print('Fitting model')
    seeds.fit_model()

    print("How'd we do?")
    seeds.evaluate_model()