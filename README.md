# Diagnosis of Viral Pneumonia:
## Building CNN from Scratch for Pneumonia Diagnosis by Classifying Chest X-Ray Images
![](https://github.com/peimani/Project4/blob/bf944f4a107dff66210a24e3d63c6fc026cb265c/Pics%20Proj4/Screen%20Shot%202020-12-16%20at%209.05.07%20AM.png)


### Motivation
This Vision Project can have many uses, one of which is it's use in the medical field. Using Artificial Intelligence can help the healthcare industry by providing practitioners with faster diagnosis. These tools can also help providers by pointing them to the right direction for a more accurate diagnosis.  There can also be onsite as well as remote diagnosis with labels already provided.  This could not only speed up the diagnosis but also save money by providing an automated report which may replace the radiologists function.


### Introduction
In this project I will set up an image classifier which will look at chest X-rays and classify them as Normal or Pneumonia. Using tools such as Convolutional Neural Network, Keras, and Tensor Flow; this program will iterate through images and classify them.  I will analyze this after the model is implemented to see how accurate they are in classification. 


## Objectives
My objective is to create a program which will allow medical professionals to take an X-ray and have a diagnosis on-site. 

For this module's final project, I have the choice of the study:

Image Classification with Deep Learning
I pick up one chest x-ray image classification problem as my project, and I plan to tackle it by means of deep learning, because I consider Portfolio Depth.

**The Steps Taken:**

1. Load images from a hierarchical file structure using an image datagenerator
2. Apply data augmentation to image files before training a neural network
3. Build a CNN using Keras
4. Visualize and evaluate the performance of CNN models
5. Load saved Keras models
6. Use Keras methods to visualize activation functions in CNNs
7. Take advantage of pretrained networks
8. Study how pre-trained neural networks benefit feature extraction
9. Understand what "freezing" and "unfreezing" a layer means in a neural network
10. Implement feature engineering and fine tuning on a pre-trained model
11. Use Keras to adapt a pretrained CNN
12 Loading Data for Image Classification with Deep Learning

 The data for this project concerns lung xray images for pneumonia. The original dataset is from Kaggle. I have downloaded the entire dataset for the sake of model training in order to design various architectures and evaluate their performaces as well by fitting to data. ⏰

To build a deep neural network that trains on a large dataset for classification is a non-trivial task. In this case, I utilize x-ray images of pediatric patients in order to identify whether or not they have pneumonia. The entire dataset comes from Kermany et al. on Mendeley, although there is also a version on Kaggle that may be easier to use.

This task is to:

    Build a model that can classify whether a given patient has pneumonia, given a chest x-ray image.
To speed up image pre-processing, 1024x1024 images were downsized. All images are categorized into two groups: NORMAL and PNEUMONIA.

The Number of DataSet Images:

![](https://github.com/peimani/Project4/blob/master/Pics%20Proj4/Screen%20Shot%202020-12-16%20at%207.36.10%20AM.png)




## Loading Models for Visualizing Intermediate Activations of every Deep Learning Model
Deep learning is extremely powerful and is helping to lead the advancement of many AI tasks. That said, deep learning is often criticized for having a lot of black box algorithms in that the components of the model itself are difficult to interpret. In the case of CNNs and image recognition, this is actually not true at all! In this lesson, we explored how to visualize the intermediate hidden layers within your CNN to uncover what sorts of features your deep network is uncovering through some of the various filters. With that, you'll gain interesting insights and knowledge as to how your CNN is seeing the world.

Once the CNN is built we load pretrained models and visualize learned features. It is possible to visualize all channels from each layer in a CNN. The pretrained models were saved and investigated. The difficulty was the quality and size of the training data. This is why it helped to use a pretrained model with a similar dataset. 

Pretrained models are adapted for classifying `PNEUMONIA` or `NORMAL` problem scenario that I've worked on so far!

### Tuning Models for the Optimization of Chosen Metrics:
-Augmentation
-Early stopping: through varying epochs, the first point at which the local minimal loss appears can be identified.
-Trainging optimizers
-Regularizers: the addition of first- and second-order regularization terms into the cost function to smoothen the variation of both cost and accuracy                    with time
-Dropout
-Activation functions
-Batch size: see below
-Learning rate: see below
-Comparison of Model Performances

## Conclusion
Neural Networks can be challenging to design, but are able to read x-ray images and distinguish certain disorders. My recommendations:

 * Google Colab is a good place to write your code because it provides the extra power needed to run the models which are of large size and quality.
 * Keras features are very useful. **fit_generator** was used for training Keras a model using Python data generators. **ImageDataGenerator** was also used for real time data augmentation.
 * Using PreTrained Networks will help tune a huge number of images in the training data.
      -- As such, adapting a pretrained model that was trained on a larger dataset can lead to a stronger model when you have limited training data.
 * For more information on using Keras and how to handle small amounts of data this is a good reference: [Keras Blog](https://https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html)

![](https://github.com/peimani/Project4/blob/master/Pics%20Proj4/Screen%20Shot%202020-12-16%20at%209.28.19%20AM.png)
