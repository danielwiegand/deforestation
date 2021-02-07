import os
import random
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing import image
from matplotlib import pyplot as plt

IMAGE_PATH_TRAIN = "../data/images/train-jpg/"
IMAGE_PATH_TEST = "../data/images/test-jpg/"
y_labels = pd.read_csv("../data/y_labels/train_v2.csv")

train_dir = os.listdir(IMAGE_PATH_TRAIN)
test_dir = os.listdir(IMAGE_PATH_TEST)
len(train_dir), y_labels.shape, len(test_dir)

def show_random_image():
    img = image.load_img(IMAGE_PATH_TRAIN + random.choice(train_dir))
    return img

# Overview of some images
for i in range(25):
    plt.subplot(5, 5, i+1)
    plt.imshow(show_random_image())
    plt.axis('off')

# Examine a single image
img_array = image.img_to_array(img).astype(np.uint8)
img_array.shape

# Examine y labels
y_labels_split = y_labels.merge(y_labels["tags"].str.split(" ", expand = True), left_index = True, right_index = True).drop("tags", axis = 1)

y_labels["tags"].value_counts()
y_labels["tags"].value_counts().plot(kind = "bar")
plt.xticks(rotation = 90)

y_labels["tags"].value_counts()[0:30].plot(kind = "bar")

y_labels_split.drop("image_name", axis = 1).unstack().dropna().value_counts()
