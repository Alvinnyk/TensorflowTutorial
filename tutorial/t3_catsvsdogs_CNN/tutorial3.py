import tensorflow as tf
import numpy as np
import os
from tensorflow.keras.callbacks import TensorBoard
import time
from tutorial.t3_catsvsdogs_CNN.config import *

'''
    ========== Tutorial 3 ==========
    Convolutional Neural Network to classify cats and dogs

    Cats and dogs dataset from microsoft contains ~25k images
    Each image is of different sizes and dimensions
    Refer to data_processor_catsvsdogs for data processing
    
    This script trains the model and saves it in model/

    Use this command on terminal to view tensorboard
        tensorboard --logdir logs/
'''


def train_model(img_data, img_label):
    """
        Conv2D layer requires an input shape of (num_of_images, x_shape, y_shape, channels)
        num_of_images:      number of images
        x_shape:            the width of the image
        y_shape:            the height of the image
        channels:           number of channels (colours) i.e. 3 channels for RGB, 1 channel for grayscale

        Conv2D( filter, kernel_size, padding, activation)
        filter:             the number filters to use
        kernel_size         the 2D dimensions of each filter
        padding:            valid:      no padding
                            same:       pad to make output size same as input size
        activation:         to normalize the output neuron

        MaxPooling2D( pool_size, strides, padding)
        pool_size:          specifies size of the pooling window
        strides:            specifies strides of the pooling action
        padding:            to ensure output size is the same
    """
    model_name = "cvd_CNN_{}x{}_{}_{}".format(IMG_SIZE, IMG_SIZE, "128-128-64", int(time.time()))
    model = tf.keras.Sequential([
        # input layer with img_data.shape
        tf.keras.layers.Input(shape=img_data[0].shape),

        # First Conv2D layer
        tf.keras.layers.Conv2D(128, kernel_size=(3, 3), padding="same", activation="relu"),
        #tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2, padding="valid"),

        # second Conv2D layer
        tf.keras.layers.Conv2D(128, kernel_size=(3, 3), padding="same", activation="relu"),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2, padding="valid"),

        # third Conv2D layer
        tf.keras.layers.Conv2D(64, kernel_size=(3, 3), padding="same", activation="relu"),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2, padding="valid"),

        # flatten out the 2D layer and feed into a dense layer of 32 neurons
        tf.keras.layers.Flatten(),
        # tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(2, activation="softmax")
    ])

    model.compile(optimizer="adam",
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                  metrics=["accuracy"])

    # Tensorboard object for callbacks
    tensorboard = TensorBoard(log_dir="logs\{}".format(model_name))

    model.fit(img_data, img_label, batch_size=64, epochs=8, validation_split=0.2, callbacks=[tensorboard])

    print(model.summary())

    model.save("model/{}.h5".format(model_name))


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

    # Trains the model and saves it
    train_model(img_data, img_label)
