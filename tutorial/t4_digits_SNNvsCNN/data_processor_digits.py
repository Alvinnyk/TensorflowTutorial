import tensorflow as tf
import numpy as np

'''
   Downloads, processes and saves digit dataset from MNIST 
'''
if __name__ == "__main__":

    try:
        # Loads data from MNIST
        print("Downloading digits dataset from MNIST..")
        mnist = tf.keras.datasets.mnist
        (train_data, train_label), (test_data, test_label) = mnist.load_data()

        img_data = np.concatenate((train_data, test_data))
        img_label = np.concatenate((train_label, test_label))

        img_data = img_data/255.0

        # Saves data locally for future use
        print("Saving dataset to /data")
        np.save("data/img_data", img_data)
        np.save("data/img_label", img_label)

        print("Data processing completed.")

    except:
        print("Data processing failed.")
        exit(0)

