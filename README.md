# AMLS_21-22_SN18071511

## Introduction
We have been given a labelled collection of 3000 training and validation MRI scans plus 200 testing MRI scans to be used as datasets for classifying brain tumors. There are two separate tasks given: Task A entails a binary classification in which the brain image is classified between having a tumour or not, Task B entailss a multiclass classification in which the brain image is classified between glioma tumour, meningioma tumour, pituitary tumour or no tumour. For Task A, SVM and CNN models have been utilized whereas only CNN was used for Task B.  

## Description
The 'dataset' folder contains all the initial data that has been provided for training and validation. Inside, there is a folder containing all the images and a '.csv' file containing all the relevant labels. 'test' folder is similar but contains the testing dataset. The main folder contains the preprocessing document 'data_acquisition.py' as well as documents for the models 'TaskA-SVM', 'TaskA-CNN', 'TaskB-CNN'.  

## Role of the Each File
'dataset' and 'test' folders are for the datasets.  
'data_acquisition.py' contains the image preprocessing in which the images are converted to arrays with mathematical labels for further coding.  
'TaskA-SVM', 'TaskA-CNN', 'TaskB-CNN' contain the main body of the code for machine learning models tackling the tasks.  
'README.md' contains the preliminary information before running the ccodes present in the folder.  

## How to run
For running, please download the folder onto your pc (if you are on MACOS make sure to change the path divider inside 'data_acquisition.py' for correct pathing). Make sure the dataset folders ('dataset' and 'test') are in the same folder as the code. Open up Jupyter Notebook through Anaconda and navigate to the said folder. Open and run the code in order 'TaskA-SVM', 'TaskA-CNN', 'TaskB-CNN'.  

## Packages required
numpy  
matplotlib  
sklearn  
time  
tensorflow  
keras  
skimage  
