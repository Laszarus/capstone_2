import numpy as np
import scipy.io

from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

from load_data import ImageData

class AppleseedCNN(object):
    def __init__(self, mat, sample_size=2000):
        # .mat loads im as a dictionary
        self.mat = mat
        
        # assign values (images) to variables
        self.x_train = mat.get('train_x')
        self.y_train = mat.get('train_y')
        self.x_test = mat.get('test_x')
        self.y_test = mat.get('test_y')
        self.ann = mat.get('annotations')

        # cut off that pesky NIR layer
        self.x_train = self.x_train[:,:,0:3,0:sample_size]
        self.x_test = self.x_test[:,:,0:3,0:sample_size]

        # reshape to what Conv2D prefers
        self.x_train = self.x_train.reshape(self.x_train.shape[3], self.x_train.shape[0], 
                                            self.x_train.shape[1], self.x_train.shape[2])
        self.x_test = self.x_test.reshape(self.x_test.shape[3], self.x_test.shape[0], 
                                          self.x_test.shape[1], self.x_test.shape[2])
        self.y_train = self.y_train.T
        self.y_test = self.y_test.T

        # create a locally manageable dataset
        self.x_train_play = self.x_train[0:sample_size,:,:,:]
        self.y_train_play = self.y_train[0:sample_size,:]
        self.x_test_play = self.x_test[0:sample_size,:,:,:]
        self.y_test_play = self.y_test[0:sample_size,:]

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

        # now start a typical neural network
        model.add(Dense(32)) # (only) 32 neurons in this layer, really?   KEEP
        model.add(Activation('relu'))

        model.add(Dropout(0.175)) # zeros out some fraction of inputs, helps prevent overfitting

        model.add(Dense(nb_classes)) # 10 final nodes (one for each class)  KEEP
        model.add(Activation('softmax')) # softmax at end to pick between classes 0-9 KEEP
        
        # many optimizers available, see https://keras.io/optimizers/#usage-of-optimizers
        # suggest you KEEP loss at 'categorical_crossentropy' for this multiclass problem,
        # and KEEP metrics at 'accuracy'
        # suggest limiting optimizers to one of these: 'adam', 'adadelta', 'sgd'
        model.compile(loss='categorical_crossentropy',
                    optimizer='adam',
                    metrics=['accuracy'])
        self.model = model

    def fit_model(self,input_data='play'):
        if input_data == 'play':
            self.model.fit(self.x_train_play, self.y_train_play, batch_size=batch_size, epochs=nb_epoch,
            verbose=1, validation_data=(self.x_test_play, self.y_test_play))
        elif input_data == 'full':
            self.model.fit(self.x_train, self.y_train, batch_size=batch_size, epochs=nb_epoch,
            verbose=1, validation_data=(self.x_test, self.y_test))
    
    def evaluate_model(self,input_data='play'):
        if input_data == 'play':
            score = self.model.evaluate(self.x_test_play, self.y_test_play, verbose=0)
            print('Test score:', score[0])
            print('Test accuracy:', score[1]) # the important one

        elif input_data == 'full':
            score = self.model.evaluate(self.x_test, self.y_test, verbose=0)
            print('Test score:', score[0])
            print('Test accuracy:', score[1]) # the important one


if __name__ == "__main__":

    print('Creating class')
    mat = scipy.io.loadmat('data/sat-6-full.mat')
    apples = AppleseedCNN(mat)

    print("Initializing parameters")
    batch_size = 10  # number of training samples used at a time to update the weights
    nb_classes = 6   # number of output possibilites: [0 - 9] KEEP
    nb_epoch = 4    # number of passes through the entire train dataset before weights "final"
    img_rows, img_cols = 28, 28  # the size of the NAIP images
    input_shape = (img_rows, img_cols, 3)  # 4 channel image input (RGB) 
    nb_filters = 10 # number of convolutional filters to use
    pool_size = (2, 2) # pooling decreases image size, reduces computation, adds translational invariance
    kernel_size = (4, 4) # convolutional kernel size, slides over image to learn features

    print('Building model')
    apples.define_model(nb_filters, kernel_size, input_shape, pool_size)

    print('Fitting model')
    apples.fit_model()

    print("How'd you do?")
    apples.evaluate_model()