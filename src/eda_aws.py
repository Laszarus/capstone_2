import matplotlib.pyplot as plt 
import numpy as np
import scipy.io
from skimage import io
from load_data import ImageData
from PIL import Image
'''
def class_distribution_bar_graph(y_list=data.y_train_play_cl):
    
    # EDA tool to display number of images per class in the dataset. 

    # Inputs: 
    # y_list = ImageData.y_train_cl, ImageData.y_test_cl,
    #             ImageData.y_train_sample_cl, ImageData.y_test_sample_cl  

    # Returns: Bar graph displaying number of images per class      
    
    # count and sorts number of images in each class
    unique_elements, counts_elements = np.unique(y_list, return_counts=True)
    sorting = np.argsort(unique_elements)
    counts_ordered = counts_elements[sorting]
    classes_ordered = unique_elements[sorting]
    y_max = counts_ordered.max()
    y_interval = int(y_max/20)

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
    plt.show()
'''
def image_classification_samples(folder='x_train'):
    '''
    Inputs: folder; String type. Folder name of which population you want
            to pull samples from.
    Returns: 6x (1x6) plots displaying random sample images from each
             class of land cover. 
    '''
    
    cols = [0, 1, 2, 3, 4, 5]

    fig, ax = plt.subplots(1,6,figsize=(15,15))
    for i in cols: 
        plt.grid(b=None, which='both')
        building = io.imread('data/{}/{}/0_building/building{}.png'.format(mode,folder,np.random.randint(0,40)))
        ax[i].imshow(building)
        fig.suptitle('building',x=0.03, y=.5)

    fig, ax = plt.subplots(1,6,figsize=(15,15))
    for i in cols: 
        barren = io.imread('data/{}/{}/1_barren_land/barren{}.png'.format(mode,folder,np.random.randint(0,40)))
        ax[i].imshow(barren)
        fig.suptitle('barren',x=0.03, y=.5)

    fig, ax = plt.subplots(1,6,figsize=(15,15))
    for i in cols: 
        tree = io.imread('data/{}/{}/2_tree/tree{}.png'.format(mode,folder,np.random.randint(0,40)))
        ax[i].imshow(tree)
        fig.suptitle('tree',x=0.03, y=.5)

    fig, ax = plt.subplots(1,6,figsize=(15,15))
    for i in cols: 
        grass = io.imread('data/{}/{}/3_grassland/grassland{}.png'.format(mode,folder,np.random.randint(0,40)))
        ax[i].imshow(grass)
        fig.suptitle('grass',x=0.03, y=.5)

    fig, ax = plt.subplots(1,6,figsize=(15,15))
    for i in cols: 
        road = io.imread('data/{}/{}/4_road/road{}.png'.format(mode,folder,np.random.randint(0,40)))
        ax[i].imshow(road)
        fig.suptitle('road',x=0.03, y=.5)

    fig, ax = plt.subplots(1,6,figsize=(15,15))
    for i in cols: 
        water = io.imread('data/{}/{}/5_water/water{}.png'.format(mode,folder,np.random.randint(0,40)))
        ax[i].imshow(water)
        fig.suptitle('water',x=0.03, y=.5)
    
    fig.savefig('images/img_class_samples_aws.png')
    plt.show()

def plot_confusion_matrix(y_true, y_pred, 
                          classes,
                          normalize=False,
                          title='Dazed and Confusioned',
                          cmap=plt.cm.YlOrRd):

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    print(cm)

    fig, ax = plt.subplots(figsize=(8,8))
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax
    fig.savefig('images/confusion_aws.png')
    plt.show()

def create_accuracy_loss(self, figloc):

    # This will plot the accuracy and loss plots for the model

    fig, ax = plt.subplots(1, 2, figsize=(12, 9))
    ax[0].plot(self.df['accuracy'], lw=3, marker='.')
    ax[0].plot(self.df['val_accuracy'], lw=3, marker='.')
    ax[0].set_title('Model accuracy')
    ax[0].set_ylabel('Accuracy')
    ax[0].set_xlabel('Epoch')
    ax[0].legend(['Train', 'Test'], loc='upper left')

    ax[1].plot(self.df['loss'], lw=3, marker='.')
    ax[1].plot(self.df['val_loss'], lw=3, marker='.')
    ax[1].set_title('Model loss')
    ax[1].set_ylabel('Loss')
    ax[1].set_xlabel('Epoch')
    ax[1].legend(['Train', 'Test'], loc='upper left')
    a = self.score_accuracy
    ls = self.score_loss
    fig.suptitle(f'Model V{self.waldo.version} Loss:{ls} Acc:{a} (on holdout)',
                    fontsize=18)
    plt.savefig(figloc)

def pred_building_road_true():
    pred_0_list = []
    pred_0 = np.argwhere(y_pred == 0)
    pred_0 = pred_0.tolist()
    for x in range(len(pred_0)):
        pred_0_list.append(pred_0[x][0])

    true_4_list = []
    true_4 = np.argwhere(y_true == 4)
    true_4 = true_4.tolist()
    for x in range(len(true_4)):
        true_4_list.append(true_4[x][0])
    
    pred_0_set = set(pred_0_list)
    true_4_set = set(true_4_list)
    
    mistakes = (pred_0_set & true_4_set)
    mistakes = list(mistakes)
    
    fig, ax = plt.subplots(1,5,figsize=(15,15))
    for i in range(5):
        data = x_test_sample[mistakes[i],:,:,:]
        img = Image.fromarray(data,'RGB')
        ax[i].imshow(img)
        fig.suptitle('Predicted building, actually road: 11/250',x=0.5, y=.4)
        fig.savefig('images/pred_building_road_true_aws.png')

def pred_barren_grass_true():
    pred_1_list = []
    pred_1 = np.argwhere(y_pred == 1)
    pred_1 = pred_1.tolist()
    for x in range(len(pred_1)):
        pred_1_list.append(pred_1[x][0])

    true_3_list = []
    true_3 = np.argwhere(y_true == 3)
    true_3 = true_3.tolist()
    for x in range(len(true_3)):
        true_3_list.append(true_3[x][0])
    
    pred_1_set = set(pred_1_list)
    true_3_set = set(true_3_list)
    
    mistakes = (pred_1_set & true_3_set)
    mistakes = list(mistakes)

    fig, ax = plt.subplots(1,5,figsize=(15,15))
    for i in range(5):
        data = x_test_sample[mistakes[i],:,:,:]
        img = Image.fromarray(data,'RGB')
        ax[i].imshow(img)
        fig.suptitle('Predicted barren, actually grass: 37/250',x=0.5, y=.4)
        fig.savefig('images/pred_barren_grass_true_aws.png')

def pred_grass_tree_true():
    pred_3_list = []
    pred_3 = np.argwhere(y_pred == 3)
    pred_3 = pred_3.tolist()
    for x in range(len(pred_3)):
        pred_3_list.append(pred_3[x][0])

    true_2_list = []
    true_2 = np.argwhere(y_true == 2)
    true_2 = true_2.tolist()
    for x in range(len(true_2)):
        true_2_list.append(true_2[x][0])
    
    pred_3_set = set(pred_3_list)
    true_2_set = set(true_2_list)
    
    mistakes = (pred_3_set & true_2_set)
    mistakes = list(mistakes)

    fig, ax = plt.subplots(1,5,figsize=(15,15))
    for i in range(5):
        data = x_test_sample[mistakes[i],:,:,:]
        img = Image.fromarray(data,'RGB')
        ax[i].imshow(img)
        fig.suptitle('Predicted grass, actually tree: 29/250',x=0.5, y=.4)
        fig.savefig('images/pred_grass_tree_true_aws.png')

def pred_road_building_true():
    pred_4_list = []
    pred_4 = np.argwhere(y_pred == 4)
    pred_4 = pred_4.tolist()
    for x in range(len(pred_4)):
        pred_4_list.append(pred_4[x][0])

    true_0_list = []
    true_0 = np.argwhere(y_true == 0)
    true_0 = true_0.tolist()
    for x in range(len(true_0)):
        true_0_list.append(true_0[x][0])
    
    pred_4_set = set(pred_4_list)
    true_0_set = set(true_0_list)
    
    mistakes = (pred_4_set & true_0_set)
    mistakes = list(mistakes)

    fig, ax = plt.subplots(1,5,figsize=(15,15))
    for i in range(5):
        data = x_test_sample[mistakes[i],:,:,:]
        img = Image.fromarray(data,'RGB')
        ax[i].imshow(img)
        fig.suptitle('Predicted grass, actually tree: 29/250',x=0.5, y=.4)
        fig.savefig('images/pred_road_building_true_aws.png')

if __name__ == "__main__":   
    mat = scipy.io.loadmat('data/sat-6-full.mat')
    data = ImageData(mat)

    mode = 'RGB'
    sample_data = np.load('data/{}/{}_aws_data.npz'.format(mode,mode))

    x_train_sample = sample_data['arr_0']
    y_train_sample = sample_data['arr_1']
    x_test_sample = sample_data['arr_2']
    y_test_sample = sample_data['arr_3']

    con_matrix_data = np.load('data/RGB/aws_confusion.npz')
    y_pred = con_matrix_data['arr_0']
    y_true = con_matrix_data['arr_1']   
