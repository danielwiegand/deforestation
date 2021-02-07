import os
import random

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, fbeta_score, classification_report
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import (Activation, BatchNormalization, Conv2D, Dense, Dropout, Flatten, MaxPooling2D)
from tensorflow.keras.models import Sequential
from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing.image import (ImageDataGenerator,
                                                  img_to_array, load_img)
from PIL import Image
from timeit import default_timer as timer


# * FUNCTIONS

def show_image(path):
    img = load_img(path)
    return img

def image_from_array(array):
    img = Image.fromarray(array, 'RGB')
    return img

def show_image_predictions(generator):
    for i in range(4):
        img, labels = generator.next()
        labels = np.where(labels[0] == 1)[0]
        classes = list(generator.class_indices)
        labels = [classes[i] for i in labels]
        plt.subplot(2, 2, i+1)
        plt.imshow(img[0])
        plt.title(labels)
        plt.axis('off')

def load_labels():
    y_labels = pd.read_csv("../data/y_labels/train_v2.csv")
    y_labels["tags"] = y_labels["tags"].apply(lambda x:x.split(" "))
    y_labels["image_name"] = y_labels["image_name"].apply(lambda x: x + ".jpg")
    UNIQUE_LABELS = list(set([x for sublist in y_labels["tags"] for x in sublist]))
    
    return y_labels, UNIQUE_LABELS

def create_generator(df, directory, batch_size, shuffle, classes):
    
    datagen = ImageDataGenerator(rescale = 1./255.)
                                #    featurewise_center = True,
                                #    featurewise_std_normalization = True,
                                #    rotation_range = 20,
                                #    width_shift_range = 0.2,
                                #    horizontal_flip = True,
                                #    vertical_flip = True)

    generator = datagen.flow_from_dataframe(
        dataframe = df,
        directory = directory,
        x_col = "image_name",
        y_col = "tags",
        batch_size = batch_size,
        seed = 42,
        shuffle = shuffle,
        classes = classes,
        class_mode = "categorical",
        target_size = (256,256))
    
    return generator

def build_model(config, n_labels):
    
    m = Sequential([ 
        # Layer 1
        Conv2D(filters = 32, 
            kernel_size = (3, 3), 
            strides = (2, 2),
            padding = "same",
            input_shape = (256, 256, 3)),
        Activation(config.activation),
        # Layer 2
        Conv2D(filters = 32, 
            kernel_size = (3, 3), 
            strides = (2, 2),
            padding = "same"),
        MaxPooling2D(pool_size = (3, 3), strides = (2, 2), padding = "same"),
        Activation(config.activation),
        BatchNormalization(),
        # Layer 3
        Conv2D(filters = 64, 
            kernel_size = (3, 3), 
            strides = (2, 2),
            padding = "same"),
        Activation(config.activation),
        # Layer 4
        Conv2D(filters = 64, 
            kernel_size = (3, 3), 
            strides = (2, 2),
            padding = "same"),
        MaxPooling2D(pool_size = (3, 3), strides = (2, 2), padding = "same"),
        Activation(config.activation),
        BatchNormalization(),
        Flatten(),
        # Output Layer
        Dense(units = 512, activation = "elu"),
        Dense(units = n_labels, activation = "sigmoid"),
        ])
    
    return m