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

        # Saves data locally for future use
        print("Saving dataset to /data")
        np.save("data/train/train_data", train_data)
        np.save("data/train/train_label", train_label)
        np.save("data/test/test_data", test_data)
        np.save("data/test/test_label", test_label)

        print("Data processing completed.")

    except:
        print("Data processing failed.")
        exit(0)

