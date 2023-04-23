import tensorflow as tf
from tensorflow import keras
from keras import layers
import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
import random

img_array = cv2.imread("Training/0/Training_44626307.jpg")  # this reads the image

data_directory = "Training/"    # this is going to be our training dataset
classes = ["0", "1", "2", "3", "4", "5", "6"]   # this is the list of classes -> this has to be the exact name of the training folders

# this changes the size of the original images(48x48) to 
# ImageNet(a popular and new database of pictures) size which is (224x224)
img_size = 224
new_array = cv2.resize(img_array, (img_size, img_size))
plt.imshow(cv2.cvtColor(new_array, cv2.COLOR_BGR2RGB))
plt.show()

# this is going to read all the images and we are going to resize them 
# into imageNet size and put them in an array called training_data

training_data = []

def create_training_data():
    for category in classes:
        path = os.path.join(data_directory, category)
        class_num = classes.index(category)
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path, img))
                new_array = cv2.resize(img_array, (img_size, img_size))
                training_data.append([new_array, class_num])
            except Exception as e:
                pass

create_training_data()

# the reason why we're randomizing our data is because our
# deep learning model should not learn the sequence
random.shuffle(training_data)

X = []
Y = []

for features,label in training_data:
    X.append(features)
    Y.append(label)

X = np.array(X).reshape(-1, img_size, img_size, 3)  # converting the data into 4 dimensions

# we normalize the data
X = X/255.0

# Dividing the pixel values by 255 normalizes the data to a range between 0 and 1,
# which is useful for several reasons. 
# For example, it helps the model to converge faster during training, 
# and it also ensures that the input data is on the same scale, 
# which can improve the accuracy of the model. Additionally, 
# it helps to avoid numerical overflow or underflow issues that 
# can occur when working with large or small values.

Y = np.array(Y)

