import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import random

# set this to true to loop through all images randomly
SHUFFLE = False

# set this to true to see all wrong classifications
SHOW_WRONG = True

def loadData():
    img_data = np.load("data/img_data.npy")
    img_label = np.load("data/img_label.npy")
    return img_data, img_label


if __name__ == "__main__":

    img_data = []
    img_label = []

    try:
        print("Loading dataset...")
        img_data, img_label = loadData()
    except:
        print("Failed to load dataset.")
        exit(0)

    model_SNN = tf.keras.models.load_model("model/SNN_model.h5")
    model_CNN = tf.keras.models.load_model("model/CNN_model.h5")

    if SHUFFLE:

        img_data_CNN = img_data.reshape(-1, 28, 28, 1)
        img_data_SNN = img_data.reshape(-1, 28, 28)

        SNN_prediction_array = model_SNN.predict(img_data_SNN)
        CNN_prediction_array = model_CNN.predict(img_data_CNN)

        indexes = list(range(len(img_data)))
        random.shuffle(indexes)

        print("============================================================")
        print("     Shuffling through {} images".format(len(img_data)))
        print("============================================================")

        for index in indexes:

            SNN_prediction_label = int(np.argmax(SNN_prediction_array[index]))
            CNN_prediction_label = int(np.argmax(CNN_prediction_array[index]))
            true_label = img_label[index]

            plt.figure(figsize=(10, 4))

            plt.subplot(1, 3, 1)
            plt.imshow(img_data[index], cmap="gray_r")
            plt.xlabel("True Label: {}".format(img_label[index]))

            plt.subplot(1, 3, 2)
            plt.title("SNN: {:0.5f}%".format(100 * np.max(SNN_prediction_array[index])))
            plt.xticks(range(10))
            thisplot = plt.bar(range(10), SNN_prediction_array[index], color="#777777")
            plt.ylim([0, 1])
            thisplot[SNN_prediction_label].set_color('red')
            thisplot[true_label].set_color('blue')

            plt.subplot(1, 3, 3)
            plt.title("CNN: {:0.5f}%".format(100 * np.max(CNN_prediction_array[index])))
            plt.xticks(range(10))
            thisplot = plt.bar(range(10), CNN_prediction_array[index], color="#777777")
            plt.ylim([0, 1])
            thisplot[CNN_prediction_label].set_color('red')
            thisplot[true_label].set_color('blue')

            plt.show()

    elif SHOW_WRONG:
        img_data_CNN = img_data.reshape(-1, 28, 28, 1)
        img_data_SNN = img_data.reshape(-1, 28, 28)

        SNN_prediction_array = model_SNN.predict(img_data_SNN)
        CNN_prediction_array = model_CNN.predict(img_data_CNN)

        print("============================================================")
        print("     Showing images classified incorrectly")
        print("============================================================")

        for index in range(len(img_data)):
            SNN_prediction = SNN_prediction_array[index]
            SNN_prediction_label = int(np.argmax(SNN_prediction))

            CNN_prediction = CNN_prediction_array[index]
            CNN_prediction_label = int(np.argmax(CNN_prediction))

            true_label = img_label[index]

            if SNN_prediction_label != true_label or CNN_prediction_label != true_label:

                plt.figure(figsize=(10, 4))

                plt.subplot(1, 3, 1)
                plt.imshow(img_data[index], cmap="gray_r")
                plt.xlabel("True Label: {}".format(img_label[index]))

                plt.subplot(1, 3, 2)
                plt.title("SNN: {:0.5f}%".format(100 * np.max(SNN_prediction)))
                plt.xticks(range(10))
                thisplot = plt.bar(range(10), SNN_prediction, color="#777777")
                plt.ylim([0, 1])
                thisplot[SNN_prediction_label].set_color('red')
                thisplot[true_label].set_color('blue')

                plt.subplot(1, 3, 3)
                plt.title("CNN: {:0.5f}%".format(100 * np.max(CNN_prediction)))
                plt.xticks(range(10))
                thisplot = plt.bar(range(10), CNN_prediction, color="#777777")
                plt.ylim([0, 1])
                thisplot[CNN_prediction_label].set_color('red')
                thisplot[true_label].set_color('blue')

                plt.show()

    else:
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
            img = np.array(img_data[index]).reshape(28, 28)                     # to display image
            img_array_CNN = np.array([img_data[index].reshape(28, 28, 1)])      # to feed into CNN model
            img_array_SNN = np.array([img_data[index]])                         # to feed into SNN model

            SNN_prediction_array = model_SNN.predict(img_array_SNN)
            SNN_prediction = SNN_prediction_array[0]
            SNN_prediction_label = int(np.argmax(SNN_prediction))

            CNN_prediction_array = model_CNN.predict(img_array_CNN)
            CNN_prediction = CNN_prediction_array[0]
            CNN_prediction_label = int(np.argmax(CNN_prediction))

            true_label = img_label[index]

            plt.figure(figsize=(10, 4))

            plt.subplot(1, 3, 1)
            plt.imshow(img, cmap="gray_r")
            plt.xlabel("True Label: {}".format(true_label))

            plt.subplot(1, 3, 2)
            plt.title("SNN: {:0.5f}%".format(100 * np.max(SNN_prediction)))
            plt.xticks(range(10))
            thisplot = plt.bar(range(10), SNN_prediction, color="#777777")
            plt.ylim([0, 1])
            thisplot[SNN_prediction_label].set_color('red')
            thisplot[true_label].set_color('blue')

            plt.subplot(1, 3, 3)
            plt.title("CNN: {:0.5f}%".format(100 * np.max(CNN_prediction)))
            plt.xticks(range(10))
            thisplot = plt.bar(range(10), CNN_prediction, color="#777777")
            plt.ylim([0, 1])
            thisplot[CNN_prediction_label].set_color('red')
            thisplot[true_label].set_color('blue')

            plt.show()

