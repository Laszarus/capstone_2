import scipy.io
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

class ImageData(object):
    def __init__(self, mat, sample_size=400, mode='RGBA'):
        # mode = 'RGBA' or 'RGB'

        # .mat loads in as a dictionary
        self.mat = mat
        self.mode = mode
        self.sample_size = sample_size
        self.test_sample_size = sample_size / 4

        # assign values (images) to variables
        self.x_train = mat.get('train_x')
        self.y_train = mat.get('train_y')
        self.x_test = mat.get('test_x')
        self.y_test = mat.get('test_y')
        self.ann = mat.get('annotations')

        if self.mode == 'RGB':
            # create a locally manageable, RGB dataset with NO NIR layer
            self.x_train = self.x_train[:,:,0:3,:]
            self.x_test = self.x_test[:,:,0:3,:]

        # use class_lister to create list of target cdatalassifications for
        # train and test sets
        self.y_train_cl = self.class_lister(self.y_train)
        self.y_test_cl = self.class_lister(self.y_test)
    
    def class_lister(self, y_array):
        '''
        Input: y_train, y_test
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

    def save_train_png(self):
        # Saves train set folders with images you can look at to check out the data,
        # and a number of arrays from each class equal to self.sample_size parameters 
        b = 0
        bl = 0
        t = 0
        g = 0
        r = 0
        w = 0
        x_train_balanced = []
        y_train_cl_idx = []

        for i in range(len(self.y_train_cl)):
            data = self.x_train[:,:,:,i]
            img = Image.fromarray(data,self.mode)
        
            if self.y_train_cl[i] == 0 and b < (self.sample_size / 6):
                img.save("data/{}/x_train/0_building/building{}_{}.png".format(self.mode,b,i))
                x_train_balanced.append(data)
                y_train_cl_idx.append(i)
                b +=1
            elif self.y_train_cl[i] == 1 and bl < (self.sample_size / 6):
                img.save("data/{}/x_train/1_barren_land/barren{}_{}.png".format(self.mode,bl,i))
                x_train_balanced.append(data)
                y_train_cl_idx.append(i)
                bl+=1
            elif self.y_train_cl[i] == 2 and t < (self.sample_size / 6):
                img.save("data/{}/x_train/2_tree/tree{}_{}.png".format(self.mode,t,i))
                x_train_balanced.append(data)
                y_train_cl_idx.append(i)
                t+=1
            elif self.y_train_cl[i] == 3 and g < (self.sample_size / 6):
                img.save("data/{}/x_train/3_grassland/grassland{}_{}.png".format(self.mode,g,i))
                x_train_balanced.append(data)
                y_train_cl_idx.append(i)
                g+=1
            elif self.y_train_cl[i] == 4 and r < (self.sample_size / 6):
                img.save("data/{}/x_train/4_road/road{}_{}.png".format(self.mode,r,i))
                x_train_balanced.append(data)
                y_train_cl_idx.append(i)
                r+=1
            elif self.y_train_cl[i] == 5 and w < (self.sample_size / 6):
                img.save("data/{}/x_train/5_water/water{}_{}.png".format(self.mode,w,i))
                x_train_balanced.append(data)
                y_train_cl_idx.append(i)
                w+=1
            else:
                pass
        
        self.x_train_balanced = np.asarray(x_train_balanced)
        self.y_train_balanced = self.y_train.T[y_train_cl_idx]

    def save_test_png(self):
        # Saves test set directories with images you can look at to check out the data
        b = 0
        bl = 0
        t = 0
        g = 0
        r = 0
        w = 0
        x_test_balanced = []
        y_test_cl_idx = []

        for i in range(len(self.y_test_cl)):
            data = self.x_test[:,:,:,i]
            img = Image.fromarray(data,self.mode)
            
            if self.y_test_cl[i] == 0 and b < (self.test_sample_size / 6):
                img.save("data/{}/x_test/0_building/building{}_{}.png".format(self.mode,b,i))
                x_test_balanced.append(data)
                y_test_cl_idx.append(i)
                b +=1
            elif self.y_test_cl[i] == 1 and bl < (self.test_sample_size / 6):
                img.save("data/{}/x_test/1_barren_land/barren{}_{}.png".format(self.mode,bl,i))
                x_test_balanced.append(data)
                y_test_cl_idx.append(i)
                bl+=1
            elif self.y_test_cl[i] == 2 and t < (self.test_sample_size / 6):
                img.save("data/{}/x_test/2_tree/tree{}_{}.png".format(self.mode,t,i))
                x_test_balanced.append(data)
                y_test_cl_idx.append(i)
                t+=1
            elif self.y_test_cl[i] == 3 and g < (self.test_sample_size / 6):
                img.save("data/{}/x_test/3_grassland/grassland{}_{}.png".format(self.mode,g,i))
                x_test_balanced.append(data)
                y_test_cl_idx.append(i)
                g+=1
            elif self.y_test_cl[i] == 4 and r < (self.test_sample_size / 6):
                img.save("data/{}/x_test/4_road/road{}_{}.png".format(self.mode,r,i))
                x_test_balanced.append(data)
                y_test_cl_idx.append(i)
                r+=1
            elif self.y_test_cl[i] == 5 and w < (self.test_sample_size / 6):
                img.save("data/{}/x_test/5_water/water{}_{}.png".format(self.mode,w,i))
                x_test_balanced.append(data)
                y_test_cl_idx.append(i)
                w+=1
            else:
                pass
        
        self.x_test_balanced = np.asarray(x_test_balanced)
        self.y_test_balanced = self.y_test.T[y_test_cl_idx]

    def save_data(self):
        if self.mode == 'RGBA':
            # create a local subset to load from when running model sript
            np.savez('data/RGBA/RGBA_balanced_data', self.x_train_balanced, 
                                                     self.y_train_balanced,
                                                     self.x_test_balanced, 
                                                     self.y_test_balanced)
        elif self.mode == 'RGB':
            # create a local subset to load from when running model sript
            np.savez('data/RGB/RGB_balanced_data', self.x_train_balanced, 
                                                   self.y_train_balanced,
                                                   self.x_test_balanced, 
                                                   self.y_test_balanced)

if __name__ == "__main__":
    # Be sure to set 'mode' parameter before running script
    seeds = ImageData(mat = scipy.io.loadmat('data/sat-6-full.mat'), 
                     mode='RGB',
                     sample_size = 400)
    
    print("Saving {} training images...".format(seeds.mode))
    seeds.save_train_png()

    print("Saving {} testing images...".format(seeds.mode))
    seeds.save_test_png()

    print("Saving {} arrays...".format(seeds.mode))
    seeds.save_data()

    print("Done!")



