# NeuralNets

As a software engineer, you are being tasked to train the robots to identify their location based on the images of the environment. There are 5 locations that need to be identified. The locations are One-person office/Bar- bara’s office (label 0), Corridor (label 1), Two-persons office/Elin’s office (label 2), Kitchen (label 3) and Printer area (label 4). After a great amount of research, you come to the conclusion that deep learning is the most versatile approach to this problem. You decide to use 1) a feed-forward neural network and 2) and convolutional neural network (CNN) in your pursuit of the best indoor scene classifier!

Dataset:
We will be using an edited version of the INDECS Database taken from the KTH Royal Institute of Tech- nology. The database consists of several sets of pictures taken in five rooms under various illumination and weather conditions at different periods of time. Each image has a size of 64 × 85 × 3 in the RGB scale. A more detailed explanation of the data set can be found here: https://www.nada.kth.se/cas/INDECS. Please have a look at the website and explore the given information about the dataset.
To make it easier for you, we’ve already downloaded the data set and stored the images and their labels in numpy matrices, available in files stored in the data folder.
Pytorch:
We will use Pytorch, one of the most popular deep learning frameworks. In this part, you will need to read and understand our Pytorch tutorial before starting to use it. After fully understanding the tutorial code, you should be able to implement the simple feed-forward networks and convolutional neural networks using Pytorch. To help debug your code and ensure that the network is learning, a useful suggestion is to plot the loss versus training epoch to check if the loss decreases during training. You can use the Pytorch tutorial as your reference, and create a new python file to implement the following tasks.
Default Parameters:
As your default parameters, you should use:
• Adagrad as the optimizer
• 0.001 as the learning rate
• 64 as the batch size
• 40 as the maximum number of epochs • Cross-entropy loss as the loss function
Data Visualization:
Before starting with the experiments, it is important to visualize the pictures that lie in each class. Seeing the differences in each class’ pictures enables you to understand the task of the classification better. Then, you can decide on the model’s structure that can take advantages of such differences to perform better in the task. Load the data from the files in the data folder, and examine the structure of the loaded data. Visualize 6 pictures per class in the function data visualization. You will have 30 pictures in total because we have 5 classes in this classification task.
Experiment 1: Baseline Feed-Forward Neural Network [10 points]
In this part, you will implement a one-hidden-layer neural network, using the architecture shown in the table below. The original image size is 64 × 85 × 3, so you need to reshape the input image as a 16320 × 1 vector and feed it into the network. The hidden layer has 2000 neurons with ReLu activation functions. The last layer outputs a 5-element vector, representing each class. Plot the training accuracy and loss of your
 4
feed forward neural network over 40 epochs in your report. Report your final validation accuracy and all information needed to reconstruct your experiment.
Table of the structure and parameters:
Note you should complete Experiment 1 by programming the class FeedForwardNN in the skeleton python file provided.

Experiment 2: Baseline Convolutional Neural Network 
Implement the CNN architecture shown in the table below. Plot the training accuracy and loss of your CNN over 40 epochs in your report. Report your final validation accuracy and all information needed to reconstruct your experiment. Note you should complete Experiment 2 by programming the class ConvolutionalNN in the skeleton python file provided.
Table of the structure and parameters:

Experiment 3: Image Preprocessing 
In this part, you will explore how image pre-processing can play an important role in the performance of a convolutional neural network. First, instead of using the raw images, you should normalize images before training. Specifically, do the following:
Take each image and normalize pixel values in each of the RGB channel by subtracting its mean and dividing by the standard deviation. For example, if you are normalizing the red channel for an image, then for each of the red pixel values RPi, you should compute:
RPi − mean(RPi) std(RPi )
Similarly normalize the blue and green channel of each image. Note you should complete Experiment 3 by programming the function normalize image() in the skeleton python file provided. You will then have to train your baseline convolutional neural network from Experiment 2 using these normalized images. Plot the training accuracy, validation accuracy and training loss of your baseline convolutional neural network over 40 epochs with the raw images and normalized images on the same graph in your report. Report your final validation accuracy and all information needed to reconstruct your experiment. Describe the graph by comparing and contrasting the performance of these methods over validation accuracy, training accuracy, and loss over 40 epochs.

Experiment 4: Hyper-parameterization 
Hyper-parameter tuning is a very important procedure when training a neural network. In this part, you will change different hyper-parameters for one of the models in Experiment 1 or 2 (your choice). You can change anything you want in this experiment and that includes number of hidden layers, learning rate,
     Layer Hyperparameters
      Fully Connected1 Out channels = 2000. ReLu activation functions
     Fully Connected2 Out channels = 5.
      Layer Hyperparameters
      Covolution1 Kernel=(3 × 3), Out channels = 16, stride=1, padding=0. ReLu activations
     Pool1 MaxPool operation, Kernel size=(2 × 2)
     Covolution2 Kernel=(3 × 3), Out channels = 32, stride=1, padding=0. ReLu activations
     Pool2 MaxPool operation, Kernel size=(2 × 2)
     Fully Connected1 Out channels = 200. ReLu activations
     Fully Connected2 Out channels = 5.
   5

choice of optimizer, etc. The aim in this experiment is to create your most optimized neural network for this classification problem.
Implement the new hyper-parameterized neural network architecture in the OptimizedNN class as seen in the provided skeleton python file. Explain your decisions and reasoning for your choices and report your final validation accuracy of your model in the report. Report all information needed to reconstruct your experiment. After finalizing your choice of model hyper-parameters, train your model using 100% of the given training data. Evaluate your 100%-trained model by predicting the labels for the given testing data. Your implementation should output a txt file called HW4 preds.txt with only one column with one prediction per line as follows:
0
3
2
1 ...
Be very careful not to shuffle the instances; the first predicted label should correspond to the first unlabeled instance in the testing data. The number of predictions should match the number of unlabeled test instances. Plot the training accuracy and loss of your hyperparameterized neural network over 40 epochs in your report. Provide a table that summarizes the different hyperparameters that you have chosen. Explain your decisions and reasoning for your choices and report your final validation accuracy of your model in the report.
