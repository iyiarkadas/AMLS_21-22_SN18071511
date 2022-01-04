#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Importing the necessary libraries for data acquisition from MRI scans
import os
from keras.preprocessing import image

def labels_binary(base_dir):
    # takes 'base_dir' as input which specifies the path to the folder name
    # the image data can be acquired from either './dataset' or './test'
    # returns a list of the features and the labels
    
    # Specifying the path to images
    images_dir = os.path.join(base_dir,'image')
    label_filename = 'label.csv'
    
    # Creating an array of all the image paths inside the directory '/dataset/image'
    image_paths = [os.path.join(images_dir, l) for l in os.listdir(images_dir)]
    # Opening the label file and reading the lines
    label_file = open(os.path.join(base_dir, label_filename), 'r')
    label_lines = label_file.readlines()
    
    # Splitting each label with its corresponding image and putting it into a dictionary
    tumor_labels = {line.split(',')[0] : line.split(',')[1] for line in label_lines[1:]}
    
    # Getting rid of the next line identication '\n' from the data
    for k, v in tumor_labels.items():
        tumor_labels[k] = v.replace('\n', '')
        
    # Changing the value of the tumor labels to 0 if the label says 'no_tumor' and 1 otherwise 
    # as there is multiple brain tumor labels
    for k in tumor_labels:
        if 'no_tumor' in tumor_labels[k]:
            tumor_labels[k] = 0
        else:
            tumor_labels[k] = 1

    # Path divider signifies how the path is separated for different operating systems, change to '/' for macOS
    path_divider = '\\'
    
    # Checking if images_dir is an existing directory
    if os.path.isdir(images_dir):
        all_features = []
        all_labels = []
        # Running a loop for every image file in the directory
        for img_path in image_paths:
            # Getting the name of each image file
            img_file_name = img_path.split(path_divider)[-1]
            # Loading the image and converting it into an array
            # While loading image 'target_size = None' as all the images are the same size, and specifying grayscale
            img = image.img_to_array(image.load_img(img_path, target_size=None, color_mode='grayscale'))

            # If there is an image, putting its pixel information to all_features and its correspondig label to all_labels
            if img is not None:
                all_features.append(img)
                all_labels.append(tumor_labels[img_file_name])

    # return the features and labels arrays
    return (all_features, all_labels)


def labels_multiclass(base_dir):
    # Similar to labels_binary, but now one hot encoding is needed for each label

    images_dir = os.path.join(base_dir,'image')
    label_filename = 'label.csv'
    
    image_paths = [os.path.join(images_dir, l) for l in os.listdir(images_dir)]
    label_file = open(os.path.join(base_dir, label_filename), 'r')
    label_lines = label_file.readlines()

    tumor_labels = {line.split(',')[0] : line.split(',')[1] for line in label_lines[1:]}
    for k, v in tumor_labels.items():
        tumor_labels[k] = v.replace('\n', '')

    # Changing the value of the tumor labels to one hot encoding format depending whether there is no tumor,
    # meningioma tumor, glioma tumor or pituitary tumor.
    for k in tumor_labels:
        if 'no_tumor' in tumor_labels[k]:
            tumor_labels[k] = [0, 0, 0, 1]
        elif 'meningioma_tumor' in tumor_labels[k]:
            tumor_labels[k] = [1, 0, 0, 0]
        elif 'glioma_tumor' in tumor_labels[k]:
            tumor_labels[k] = [0, 1, 0, 0]
        elif 'pituitary_tumor' in tumor_labels[k]:
            tumor_labels[k] = [0, 0, 1, 0]
        else:
            tumor_labels[k] = [0, 0, 0, 0]

    path_divider = '\\'
    if os.path.isdir(images_dir):
        all_features = []
        all_labels = []
        for img_path in image_paths:
            file_name= img_path.split(path_divider)[-1]
            img = image.img_to_array(image.load_img(img_path, target_size=None, color_mode='grayscale'))            

            if img is not None:
                all_features.append(img)
                all_labels.append(tumor_labels[file_name])

    # return the features and labels arrays
    return (all_features, all_labels)


# In[ ]:




