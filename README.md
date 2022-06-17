# CS172B-MachineLeaning-and-Deep-Learning-
Final project from CS172B at University of California, Irvine

# Code:
Data_augmentation.py contains all the functions used for data augmentation procedures. Examples about how to run are shown in the same file. 

Show_data_augmentation.py contains code that was used to display some example data augmentation methods.

Helpers.py contains related helper functions including load_data for data loading, plot distribution for plotting the data distribution over the entire dataset etc. Descriptions about each function and how to use them are written clearly in each function.

Train.py contains the code of the training and validation process of CNN mode. Test. py contains code related to the testing process of CNN.

CNN.py contains code of the cnn model, which was built using Pytorch library.

Main.py contains code for the entire CNN process, from loading data, visualizing the data, building the CNN model to training, validation and testing.

The data_augmentation&CNN.ipynb contains all the data preprocessing code including data_augmentation and loading data, and CNN code(building CNN, training and validation, and testing)

Vgg.ipynb contains the vgg mode. It will call train.py and test.py to train and test the vgg model. Before running the code, it needs to be navigated to the directory containing data

TResNet.ipynb contains all the training and testing code for the TResnet model. To execute the file successfully, you need to follow the instruction on the top of the code (clone a github repository for the model structure). After solving import requirements, execute them directly and the result will be output automatically.


# Data: 
https://drive.google.com/drive/folders/1lLMTYR6O1Z91YZhtJEbuS80WOOOavfdf?usp=sharing

Original_splitted_dataset folder contains data before processing.

Our data training data is in ./train
Validation data : ./val
Test data: ./test
