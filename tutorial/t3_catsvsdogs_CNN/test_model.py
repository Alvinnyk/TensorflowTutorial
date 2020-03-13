import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os
from tutorial.t3_catsvsdogs_CNN.config import *

'''
    matplotlib application to help visualize model predictions on cats vs dogs dataset
'''

# name of model to use in prediction
MODEL_NAME = "cvd_CNN_128x128_128-128-64_1584089123.h5"


def load_data():
    img_data = np.load(os.path.join(DATA_PATH, "img_data_{}.npy".format(IMG_SIZE)))
    img_label = np.load(os.path.join(DATA_PATH, "img_label_{}.npy".format(IMG_SIZE)))
    return img_data, img_label


if __name__ == "__main__":

    img_data = []
    img_label = []

    # Loads image arrays from storage
    try:
        print("Retrieving dataset..")
        img_data, img_label = load_data()
    except:
        print("Failed to load dataset")
        exit(0)

    # Loads the saved model
    model = tf.keras.models.load_model("model/{}".format(MODEL_NAME))

    print("============================================================")
    print("     Select an index [0:{}] to see prediction".format(len(img_data)))
    print("============================================================")
    while True:
        index = input("Index: ")
        if not index.isdigit():
            print("Invalid index")
            continue
        index = int(index)
        if index >= len(img_data):
            print("Out of bounds")
            continue

        # image selected
        img = np.array(img_data[index]).reshape(IMG_SIZE, IMG_SIZE)  # to display image
        img_array = np.array([img_data[index]])  # to feed into model

        # prediction from model
        prediction_array = model.predict(img_array)
        prediction = prediction_array[0]

        # predicted and true label for img
        predicted_label = int(np.argmax(prediction))
        true_label = img_label[index]

        plt.figure(figsize=(6, 3))

        plt.subplot(1, 2, 1)
        plt.imshow(img, cmap="gray")
        if predicted_label == true_label:
            color = 'blue'
        else:
            color = 'red'
        plt.xlabel("{} {:0.2f}% ({})".format(CATEGORIES[predicted_label],
                                             100 * np.max(prediction),
                                             CATEGORIES[true_label]), color=color)

        plt.subplot(1, 2, 2)
        plt.xticks(range(2), CATEGORIES)
        thisplot = plt.bar(range(2), prediction, color="#777777")
        plt.ylim([0, 1])
        thisplot[predicted_label].set_color('red')
        thisplot[true_label].set_color('blue')

        plt.show()
