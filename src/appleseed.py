import numpy as np
import scipy.io

from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

class Appleseed(object):
    def __init__(self, mode):
        # assign train and test sets from saved subsamples
        self.mode = mode

        self.sample_data = np.load('data/{}/{}_balanced_data.npz'.format(self.mode,self.mode))

        self.x_train_sample = self.sample_data['arr_0']
        self.y_train_sample = self.sample_data['arr_1']
        self.x_test_sample = self.sample_data['arr_2']
        self.y_test_sample = self.sample_data['arr_3']

        # reshape to what Conv2D prefers--X: [samples(~6k), 28, 28, 4], y: [(samples(~1.5k), classes(6)]
        # self.x_train_sample = self.x_train_sample.reshape(self.x_train_sample.shape[3], self.x_train_sample.shape[0], 
        #                                     self.x_train_sample.shape[1], self.x_train_sample.shape[2])
        # self.x_test_sample = self.x_test_sample.reshape(self.x_test_sample.shape[3], self.x_test_sample.shape[0], 
        #                                   self.x_test_sample.shape[1], self.x_test_sample.shape[2])
        # self.y_train_sample = self.y_train_sample.T
        # self.y_test_sample = self.y_test_sample.T

        # .predict() doesn't like uint8
        self.x_train_sample = self.x_train_sample.astype('float32') / 255. # data was uint8 [0-255]
        self.x_test_sample = self.x_test_sample.astype('float32') / 255. # data was uint8 [0-255]

    def define_model(self, nb_filters, kernel_size, input_shape, pool_size):
        model = Sequential()

        # note: the convolutional layers and dense layers require an activation function
        # see https://keras.io/activations/
        # and https://en.wikipedia.org/wiki/Activation_function
        # options: 'linear', 'sigmoid', 'tanh', 'relu', 'softplus', 'softsign'

        model.add(Conv2D(nb_filters, (kernel_size[0], kernel_size[1]),
                            padding='valid', 
                            input_shape=input_shape)) #first conv. layer
        model.add(Activation('relu')) # Activation specification necessary for Conv2D and Dense layers

        model.add(Conv2D(nb_filters, (kernel_size[0], kernel_size[1]), padding='valid')) #2nd conv. layer
        model.add(Activation('relu'))

        model.add(MaxPooling2D(pool_size=pool_size)) # decreases size, helps prevent overfitting
        model.add(Dropout(0.5)) # zeros out some fraction of inputs, helps prevent overfitting

        model.add(Flatten()) # necessary to flatten before going into conventional dense layer
        print('Model flattened out to ', model.output_shape)

        # start a typical neural network
        model.add(Dense(32))
        model.add(Activation('relu'))

        model.add(Dropout(0.175)) # zeros out some fraction of inputs, helps prevent overfitting

        model.add(Dense(nb_classes)) # 6 final nodes (one for each class)
        model.add(Activation('softmax')) # softmax at end to pick between classes 0-5

        model.compile(loss='categorical_crossentropy',
                    optimizer='adam',
                    metrics=['accuracy'])
        self.model = model

    def fit_model(self):
        self.model.fit(self.x_train_sample, 
                       self.y_train_sample,
                       batch_size=batch_size, 
                       epochs=nb_epoch,
                       verbose=1, 
                       validation_data=(self.x_test_sample, self.y_test_sample))
    
    def evaluate_model(self):
        score = self.model.evaluate(self.x_test_sample, self.y_test_sample, verbose=0)
        print('{} Test score:'.format(self.mode), score[0])
        print('{} Test accuracy:'.format(self.mode), score[1]) 
    
    def check(self):
        self.pred_probs = self.model.predict(self.x_test_sample)
        print('Probabilities: {}'.format(self.pred_probs))
        self.y_pred = np.argmax(self.model.predict(self.x_test_sample), axis=1)
        print('Guesses: {}'.format(self.y_pred))
        self.y_true = np.argmax(self.y_test_sample, axis=1)
        np.savez('data/{}/confusion'.format(self.mode), self.y_pred, self.y_true)

if __name__ == "__main__":
    print('Creating class')
    apples = Appleseed(mode='RGB')
    print(apples.x_train_sample.shape)
    print(apples.y_train_sample.shape)
    print(apples.x_test_sample.shape)
    print(apples.y_test_sample.shape)

    print("Initializing parameters")
    batch_size = 10  # number of training samples used at a time to update the weights
    nb_classes = 6   # number of output possibilites: [0 - 5]
    nb_epoch = 10   # number of passes through the entire train dataset before weights "final"
    img_rows, img_cols = 28, 28  # the size of the NAIP images
    input_shape = (img_rows, img_cols, len(apples.mode))  # 28x28x3 or 4(RGB/RGBA) 
    nb_filters = 8 # number of convolutional filters to use
    pool_size = (2, 2) # pooling decreases image size, reduces computation, adds translational invariance
    kernel_size = (4, 4) # convolutional kernel size, slides over image to learn features

    print('Building model')
    apples.define_model(nb_filters, kernel_size, input_shape, pool_size)

    print('Fitting model')
    apples.fit_model()

    print("How'd we do?")
    apples.evaluate_model()

    print("What did it guess??")
    apples.check()