
![](images/banner.png)
# Appleseed 2.0
***
## Introduction
***
In 2013, The City of Denver published an [assessment](https://www.denvergov.org/media/gis/DataCatalog/tree_canopy_assessment_2013/pdf/Tree_Canopy_Assessment_2013_Final_Report.pdf) of tree coverage throughout the metropolitan area in response to the air pollution, water, and energy demand issues arising from rapid urban growth. While the report enumerated the exact number of places available to plant a tree (8 million), it lacked a useful visualization showing where these places are. This makes it more difficult for volunteer and environmental groups to take action without government involvement. In the spirit of more rapidly facilitating greener, cooler, healthier cities, this project seeks to harness the power of a convolutional neural network (CNN) to answer the question: **where can I plant some trees?**

## The Data
***
The 2013 report identified 7 different types of land cover, listed below. BSDV and grass are the obvious candidates for hosting trees, so my ultimate goal is to train a CNN to segment, and visualize the segmentation of, these types given aerial imagery.
1. Building
2. **Bare soil/dry vegetation (BSDV)**
3. Trees
4. **Grass**
5. Road
6. Water
7. Other impervious

 Before I went through the effort of hiring interns to help me capture and classify thousands of images, I figured I'd see what the world wide web had to offer. Luckily, a team of [scientists](http://bit.csc.lsu.edu/~saikat/publications/sigproc-sp.pdf) has already extracted and classified 405,000 images from the USDA's [National Agricultral Imagery Program (NAIP)](https://www.fsa.usda.gov/Internet/FSA_File/naip_2009_info_final.pdf). Images are *conveniently* classified into the following categories:
 1. Building
 2. **Barren land**
 3. Trees
 4. **Grassland**
 5. Road
 6. Water bodies

The data are also conveniently split into a four-fifths training set (324,000) and a one-fifth test set (81,000). Each image is a 28x28-pixel tile extracted from a series of about 1500 6000x7000-pixel photos taken of land throughout the state of California. These aerial photos were taken at a 1 meter ground sampling distance, meaning each pixel represents 1 meter in real space. Each image has 4 layers--red, green, blue, and near-infrared (NIR)

![](images/grass.png) ![](images/example.png)

## EDA
***
The dataset came in the form of a single MATLAB file (.mat), which when loaded into a Python is simply a dictionary with a key for each subset (train x, train y, test x, test y), and a 28x28x4xN array as the value for each key. Using Pillow's `Image.fromarray`, I saved 2000 samples from each subset. Here's an example of what each class looks like:
![](images/classes_RGBA.png)

Cool. You may notice these all have a white-ish filter on them. I attributed this to matplotlib trying to show me RGB AND NIR layers together. So when we see these images, the white layer seems to obscure the image, but I thought when the CNN read through the layers, the NIR imagery would just be additional data to help distinguish between classes. Either way, you can still pretty well identify which class each image belongs to. 

I also wanted to make sure I had enough images from each class to train on. It looks like the building and road classes are a little lacking compared to to others, but I figured I'd see how my model did first, then correct for undersampling if it seemed to be an issue.
![](images/class_distribution.png)
 
## CNN
***
I started simple:
```
Block 0:
Convolutional 2D(8 filters)
Relu activation

Block 1:
Convolutional 2D(8 filters)
Relu activation 
MaxPooling 2D(2x2)
Dropout(.5)
Flatten

Final Block:
Dense(32 neurons)
Relu activation
Dropout(.175)
Dense(6 neurons)
Softmax activation

Compiled with:
Loss: Categorical Crossentropy
Optimizer: Adam
Metrics: Accuracy
```

*Initial results:*

After fiddling with hyperparameters, the highest accuracy I could achieve was **35.1%**. 
Better then random guessing, but not ideal. 

How is the model getting confused?

Here we see the predicted probabilities of each tile being each of the 6 classifications:
```
In [12]: seeds.model.predict(seeds.x_test_play)                                                                             
Out[12]: 
array([[0.09050895, 0.15730289, 0.19790886, 0.1446698 , 0.03226209,
        0.37734747],
       [0.07537534, 0.19531024, 0.19981138, 0.13063991, 0.03852575,
        0.3603374 ],
       [0.03954737, 0.23571953, 0.16979732, 0.11483168, 0.01711163,
        0.4229924 ],
       ...,
       [0.04218896, 0.24788177, 0.16667415, 0.13383822, 0.01716879,
        0.39224815],
       [0.091229  , 0.20140629, 0.1857762 , 0.15197147, 0.05814781,
        0.3114692 ],
       [0.0446395 , 0.2253379 , 0.20204124, 0.19275795, 0.02545955,
        0.30976394]], dtype=float32)
```

And here the model's final classifications based on those probabilities:
```
In [13]: np.argmax(seeds.model.predict(seeds.x_test_play), axis=1)                                                          
Out[13]: array([5, 5, 5, ..., 5, 5, 5])
```
Okay, a lot of guesses for water, but how much?
```
In [15]: np.unique(np.argmax(seeds.model.predict(seeds.x_test_play), axis=1), return_counts=True)                           
Out[15]: (array([1, 2, 5]), array([  16,    3, 1981]))
```
To my dismay, I realized that even my humble 35% accuracy was an illusion. My model just classified every image as water, and it so happens that about 35 percent of the sample data is water. 

![](images/wrong2.jpg)

## Next steps
***
1. Restructure data
    - Try starting with binary classification (i.e. plantable/not plantable) 
        - Because pre-labeled classes will be mixed, it would be best to use the [SAT-4](https://csc.lsu.edu/~saikat/deepsat/) dataset, since it contains ~100k more images than SAT-6
    - Ensure classes are [balanced](https://towardsdatascience.com/handling-imbalanced-datasets-in-deep-learning-f48407a0e758)
        - Weight balancing
        - Synthestic Minority Oversampling Technique ([SMOTE](https://medium.com/analytics-vidhya/balance-your-data-using-smote-98e4d79fcddb))
            - Image augmentation/generation
        - Random Oversampling Examples ([ROSE](https://www.analyticsvidhya.com/blog/2016/03/practical-guide-deal-imbalanced-classification-problems/))
    - [Augment](https://towardsdatascience.com/image-augmentation-using-python-numpy-opencv-and-skimage-ef027e9898da) images to maximize applicability to images outside SAT datasets

2. Restructure model
    - Research best practices (and justifications) for aerial image [classification model architecture](https://towardsdatascience.com/semantic-segmentation-of-aerial-images-using-deep-learning-90fdf4ad780)
    - Research [hyperparameter optimization techniques](https://towardsdatascience.com/understanding-hyperparameters-optimization-in-deep-learning-models-concepts-and-tools-357002a3338a)) 

3. Retrain model with complete dataset on AWS

4. Use model with other open source aerial imagery to test applicability

5. Create classification visualization tool

6. Develop Flask app that allows user to upload an image, and returns a color-coded tree-planting map









