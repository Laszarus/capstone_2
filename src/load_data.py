import scipy.io
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

class ImageData(object):
    def __init__(self, mat, sample_size=2000):
        # .mat loads im as a dictionary
        self.mat = mat

        # assign values (images) to variables
        self.x_train = mat.get('train_x')
        self.y_train = mat.get('train_y')
        self.x_test = mat.get('test_x')
        self.y_test = mat.get('test_y')
        self.ann = mat.get('annotations')

        # create a locally manageable, RGB dataset
        self.x_train_play = self.x_train[:,:,0:3,0:sample_size]
        self.y_train_play = self.y_train[:,0:sample_size]
        self.x_test_play = self.x_test[:,:,0:3,0:sample_size]
        self.y_test_play = self.y_test[:,0:sample_size]

        # list of classifications for each array of X
        # self.y_train_cl = y_train_cl
        # self.y_test_cl = y_test_cl
        # self.y_train_play_cl = y_train_play_cl
        # self.y_test_play_cl = y_test_play_cl

        # y_train_cl = self.class_lister(self.y_train)
        # y_test_cl = self.class_lister(self.y_test)
        # y_train_play_cl = self.class_lister(self.y_train_play)
        # y_test_play_cl = self.class_lister(self.y_test_play)

        self.y_train_cl = self.class_lister(self.y_train)
        self.y_test_cl = self.class_lister(self.y_test)
        self.y_train_play_cl = self.class_lister(self.y_train_play)
        self.y_test_play_cl = self.class_lister(self.y_test_play)

    def class_lister(self, y_array):
        '''
        Input: y_train, y_test, y_train_play, y_test_play
        Returns: list of classifications (0-5) for each array in
                 the corresponding x array
        0 = building
        1 = barren land
        2 = trees
        3 = grassland
        4 = road
        5 = water
        '''

        i = 0
        class_list = []
        while i < len(y_array[1]):
            class_cat = np.where(y_array[:,i]==1)
            class_list.append(int(class_cat[0]))
            i+=1
        return class_list
    
    # def create_class_list(self):
    #     # list of classifications for each array of X
    #     self.y_train_cl = self.class_lister(self.y_train)
    #     self.y_test_cl = self.class_lister(self.y_test)
    #     self.y_train_play_cl = self.class_lister(self.y_train_play_cl)
    #     self.y_test_play_cl = self.class_lister(self.y_test_play)
        

    def save_png(self, folder, x_array, class_list, mode='RGB'):
        '''
        Inputs:
        folder = String type. Name of folder where you want images saved.
                 All child folders must follow naming convention in code. 
        x_array = x_train, x_test, x_train_play, or x_test_play
        class_list = y_train, y_test, y_train_play, or y_test_play

        Returns:
        Folders with lil images you can look at to check out the data
        '''
        b = 0
        bl = 0
        t = 0
        g = 0
        r = 0
        w = 0

        for i in range(len(class_list)):
            data = x_array[:,:,:,i]
            img = Image.fromarray(data,mode)
        
            if class_list[i] == 0:
                img.save("data/{}/0_building/building{}.png".format(folder,b))
                b +=1
            elif class_list[i] == 1:
                img.save("data/{}/1_barren_land/barren{}.png".format(folder,bl))
                bl+=1
            elif class_list[i] == 2:
                img.save("data/{}/2_tree/tree{}.png".format(folder,t))
                t+=1
            elif class_list[i] == 3:
                img.save("data/{}/3_grassland/grassland{}.png".format(folder,g))
                g+=1
            elif class_list[i] == 4:
                img.save("data/{}/4_road/road{}.png".format(folder,r))
                r+=1
            else:
                img.save("data/{}/5_water/water{}.png".format(folder,w))
                w+=1

if __name__ == "__main__":
    mat = scipy.io.loadmat('data/sat-6-full.mat')
    data = ImageData(mat)

    # # For local training
    # folder = 'x_train_play'
    # x_array = data.x_train_play
    # class_list = data.y_train_play_cl

    # For local testing
    folder = 'x_test_play'
    x_array = data.x_test_play
    class_list = data.y_test_play_cl
    
    print("Saving images...")
    data.save_png(folder, x_array, class_list,mode='RGB')

    print("Done!")



