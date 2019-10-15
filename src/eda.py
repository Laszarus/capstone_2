import matplotlib.pyplot as plt 
import numpy as np
from skimage import io
from load_data import ImageData

def class_distribution_bar_graph(class_list):
    '''
    EDA tool to display number of images per class in the dataset. 

    Inputs: 
    class_list = ImageData.y_train_cl, ImageData.y_test_cl,
                ImageData.y_train_play_cl, ImageData.y_test_play_cl  

    Returns: Bar graph displaying number of images per class      
    '''

    # use y_train data to create a list of each image array's class
    i = 0
    class_list = []
    while i < len(y[1]):
        class_cat = np.where(y[:,i]==1)
        class_list.append(int(class_cat[0]))
        i+=1
    
    # count and sorts number of images in each class
    unique_elements, counts_elements = np.unique(class_list, return_counts=True)
    sorting = np.argsort(unique_elements)
    counts_ordered = counts_elements[sorting]
    classes_ordered = unique_elements[sorting]
    y_max = counts_ordered.max()
    y_interval = y_max/20

    # plot bar graph
    N = len(classes_ordered)
    labels = ['building', 'barren land', 'trees', 'grassland', 'road', 'water']
    data = counts_ordered
    plt.style.use('fivethirtyeight')
    fig, ax = plt.subplots(figsize=(10,5))
    tickLocations = np.arange(N)
    ax.bar(tickLocations, data)
    ax.set_xticks(ticks=tickLocations)
    ax.set_xticklabels(labels)
    ax.set_yticks(range(y_max)[0::y_interval])
    ax.set_ylim((0,y_max))
    ax.set_title("Distribution of Image Classes")
    fig.tight_layout(pad=1)

def image_classification_samples(folder):
    '''
    Inputs: String type. Folder name of which population you want
            to pull samples from.
    Returns: 6x (1x6) rows displaying random sample images from each
             class of land cover. 
    '''
    
    cols = [0, 1, 2, 3, 4, 5]

    fig, ax = plt.subplots(1,6,figsize=(15,15))
    for i in cols: 
        plt.grid(b=None, which='both')
        building = io.imread('../data/{}}/0_building/building{}.png'.format(folder,np.random.randint(0,20)))
        ax[i].imshow(building)
        fig.suptitle('building',x=0.03, y=.5)

    fig, ax = plt.subplots(1,6,figsize=(15,15))
    for i in cols: 
        barren = io.imread('../data/{}}/1_barren_land/barren{}.png'.format(folder,np.random.randint(0,20)))
        ax[i].imshow(barren)
        fig.suptitle('barren',x=0.03, y=.5)

    fig, ax = plt.subplots(1,6,figsize=(15,15))
    for i in cols: 
        tree = io.imread('../data/{}}/2_tree/tree{}.png'.format(folder,np.random.randint(0,20)))
        ax[i].imshow(tree)
        fig.suptitle('tree',x=0.03, y=.5)

    fig, ax = plt.subplots(1,6,figsize=(15,15))
    for i in cols: 
        grass = io.imread('../data/{}}/3_grassland/grassland{}.png'.format(folder,np.random.randint(0,20)))
        ax[i].imshow(grass)
        fig.suptitle('grass',x=0.03, y=.5)

    fig, ax = plt.subplots(1,6,figsize=(15,15))
    for i in cols: 
        road = io.imread('../data/{}}/4_road/road{}.png'.format(folder,np.random.randint(0,20)))
        ax[i].imshow(road)
        fig.suptitle('road',x=0.03, y=.5)

    fig, ax = plt.subplots(1,6,figsize=(15,15))
    for i in cols: 
        water = io.imread('../data/{}}/5_water/water{}.png'.format(folder,np.random.randint(0,20)))
        ax[i].imshow(water)
        fig.suptitle('water',x=0.03, y=.5)