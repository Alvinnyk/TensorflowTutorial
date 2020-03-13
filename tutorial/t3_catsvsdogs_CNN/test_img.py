import numpy as np
import tensorflow as tf
import os
import cv2
import matplotlib.pyplot as plt
from tutorial.t3_catsvsdogs_CNN.config import *

'''
    This script processes images in the test_images folder and uses the model to predict if it is a cat of dog.
    You can add your own images here for testing.
    This script will predict on all images.
'''
if __name__ == "__main__":

    model_name = "cvd_CNN_128x128_128-128-64_1584087983.h5"
    model = tf.keras.models.load_model("model/{}".format(model_name))

    print("==================================================")
    print("     Model: {}".format(model_name))
    print("==================================================")

    all_images = []

    for img in os.listdir(IMG_DIR_PATH):
        img_array = cv2.imread(os.path.join(IMG_DIR_PATH, img), cv2.IMREAD_GRAYSCALE)
        img_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
        img_array = img_array / 255.0
        img_array = np.array(img_array)
        img_array = img_array.reshape(IMG_SIZE, IMG_SIZE, 1)
        all_images.append(img_array)

    all_images = np.array(all_images)
    prediction_array = model.predict(all_images)

    for i in range(len(all_images)):
        plt.figure(figsize=(7, 4))

        plt.subplot(1, 2, 1)
        plt.imshow(all_images[i].reshape(IMG_SIZE, IMG_SIZE), cmap="gray")
        plt.xlabel("Prediction: {}".format(CATEGORIES[int(np.argmax(prediction_array[i]))]), color='red')

        plt.subplot(1, 2, 2)
        plt.xticks(range(2), CATEGORIES)
        thisplot = plt.bar(range(2), prediction_array[i], color="#777777")
        plt.xlabel("Accuracy: {:0.2f}".format(100 * np.max(prediction_array[i])), color='red')
        plt.ylim([0, 1])

        plt.show()






