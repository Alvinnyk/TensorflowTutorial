import os
import cv2
import random
import numpy as np
from tutorial.t3_catsvsdogs_CNN.config import *

'''
    Processes the images
        1. converts images to image array
        2. resizes the images to IMG_SIZE x IMG_SIZE
        3. Normalizes image cells to [0:1]
        4. Shuffles the images
        5. Reshape images to (IMG_SIZE, IMG_SIZE, 1)
        6. Saves to DATA_PATH folder
        
    Download dataset from microsoft here
        https://www.microsoft.com/en-us/download/details.aspx?id=54765
'''
if __name__ == "__main__":

    all_images = []

    for category in CATEGORIES:
        class_num = CATEGORIES.index(category)  # class id of the image
        path = os.path.join(DATA_PATH, category)  # path to cats or dogs directory
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path, img),
                                       cv2.IMREAD_GRAYSCALE)  # reads in the image into an array
                # img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)    # converts BGR to RGB colour
                img_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))  # resize image to IMG_SIZE x IMG_SIZE pixels
                img_array = img_array / 255.0  # Normalizes cell to [0:1]
                all_images.append([img_array, class_num])
            except:
                # some data might be corrupted, skip those images
                print("Unable to read image: " + str(os.path.join(path, img)))

    # shuffle all the images
    random.shuffle(all_images)

    img_data = []
    img_label = []

    # split the images into its data and label
    counter = 0
    for data, label in all_images:
        img_data.append(data)
        img_label.append(label)

    # Resize the image array to follow input parameters for Conv2d layer
    img_data = np.array(img_data).reshape((-1, IMG_SIZE, IMG_SIZE, 1))
    img_label = np.array(img_label)

    # Saves processed data into storage
    np.save(os.path.join(DATA_PATH, "img_data_{}".format(IMG_SIZE)), img_data)
    np.save(os.path.join(DATA_PATH, "img_label_{}".format(IMG_SIZE)), img_label)