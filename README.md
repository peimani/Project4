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

Load images from a hierarchical file structure using an image datagenerator
Apply data augmentation to image files before training a neural network
Build a CNN using Keras
Visualize and evaluate the performance of CNN models
Load saved Keras models
Use Keras methods to visualize activation functions in CNNs
Take advantage of pretrained networks
Study how pre-trained neural networks benefit feature extraction
Understand what "freezing" and "unfreezing" a layer means in a neural network
Implement feature engineering and fine tuning on a pre-trained model
Use Keras to adapt a pretrained CNN
Loading Data for Image Classification with Deep Learning

The data for this project concerns lung xray images for pneumonia. The original dataset is from Kaggle. I have downloaded the entire dataset for the sake of model training in order to design various architectures and evaluate their performaces as well by fitting to data. ⏰

To build a deep neural network that trains on a large dataset for classification is a non-trivial task. In this case, I utilize x-ray images of pediatric patients in order to identify whether or not they have pneumonia. The entire dataset comes from Kermany et al. on Mendeley, although there is also a version on Kaggle that may be easier to use.

This task is to:

    Build a model that can classify whether a given patient has pneumonia, given a chest x-ray image.
To speed up image pre-processing, 1024x1024 images were downsized. All images are categorized into two groups: NORMAL and PNEUMONIA.

The Number of DataSet Images:

![](Pics%20Proj4/Screen Shot%202020-12-16%20at%207.36.10%20AM.png)




Loading Models for Visualizing Intermediate Activations of every Deep Learning Model
Deep learning is extremely powerful and is helping to lead the advancement of many AI tasks. That said, deep learning is often criticized for having a lot of black box algorithms in that the components of the model itself are difficult to interpret. In the case of CNNs and image recognition, this is actually not true at all! In this lesson, you will explore how you can visualize the intermediate hidden layers within your CNN to uncover what sorts of features your deep network is uncovering through some of the various filters. With that, you'll gain interesting insights and knowledge as to how your CNN is seeing the world.

Now that my own CNN has been built so that the visualization of feature maps is able to be seen. It's time to load pretrained models from files and visualize learned features systematically. Here, I'll succinctly visualize all the channels from each layer in a CNN. All of saved model files include both the model architecture and the trained weights. See the model.save() method for further details. These models were built in order to help identify patients with pneumonia. Start simply by loading them and pulling up a summary of the layers from each given pre-trained model. (To load the model use the keras.models.load_model() function.)

Loading Pretrained Models
In this lesson, you'll start to investigate how to use pretrained networks. Recall that when training neural networks, the model is tuning a huge number of weights: several to dozens at each individual layer. Often the largest limiting factor with these models is the quality and size of the training data you have at hand. As such, adapting a pretrained model from a similar problem context that was trained on a larger dataset can lead to stronger overall models when you have limited training data. For example, in image recognition, the VGG-19 network is commonly used to improve the model performance of CNNs with limited training data. VGG-19 was trained on the ImageNet dataset which contains approximately 1.2 million images. Since the initial bottom layers of a CNN pick up small details with later layers picking up larger and larger features, the initial layers of a well trained network are applicable to other problem domains. Similar pretrained networks exist for other domains such as natural language processing as well. With that, let's take a further look at how transfer learning works in detail.

Pretrained models are adapted for classifying `PNEUMONIA` or `NORMAL` problem scenario that I've worked on so far!
Tuning Models for the Optimization of Chosen Metrics:
Augmentation: see below
Early stopping: through varying epochs, the first point at which the local minimal loss appears can be identified.
Trainging optimizers: RMSprop is applied
Regularizers: the addition of first- and second-order regularization terms into the cost function to smoothen the variation of both cost and accuracy with time
Dropout: see below
Activation functions: RELU is applied
Batch size: see below
Learning rate: see below
Comparison of Model Performances:
MODEL	pixel	epoch	batch	Regularization	Dropout	Learning Rate	Augmentation	Loss	Loss	Accuracy	Accuracy
Training	Testing	Training	Testing
MLP	150	40	100	N	N	1e-4	N	0.2137	0.1858	0.9311	0.9470
CNN	150	40	100	N	N	1e-4	N	0.0283	0.1335	0.9917	0.9641
CNN	150	100	100	N	N	1e-4	Y (20% data)	0.2311	0.7139	0.8750	0.7500
CNN	150	100	100	N	N	2e-5	Y (20% data)	0.2967	0.7139	0.8840	0.7500
CNN	210	40	100	N	Y (5%)	2e-5	N	0.1282	0.1137	0.9558	0.9641
CNN	210	100	100	early stopping	N	2e-5	N	0.0706	0.0801	0.9755	0.9744
CNN	210	40	100	L1 (5e-7)	N	2e-5	N	0.1227	0.0879	0.9580	0.9726
CNN	210	40	100	L2 (5e-7)	N	2e-5	N	0.0925	0.0780	0.9661	0.9726
VGG19	150	100	5	N	N	1e-4	N	0.0057	0.6018	0.9991	0.9429
VGG19	150	100	25	N	N	2e-5	Y (20% data)	0.0479	0.2306	0.9940	0.9200
Conclusions:
xray xray

xray xray

