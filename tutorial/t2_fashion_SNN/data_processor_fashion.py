import tensorflow as tf
import numpy as np

if __name__ == "__main__":
    try:
        print("Loading fashion dataset from MNIST...")
        fashion_mnist = tf.keras.datasets.fashion_mnist
        (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

        print("Saving dataset to /data")
        np.save("data/train/train_images", train_images)
        np.save("data/train/train_labels", train_labels)
        np.save("data/test/test_images", test_images)
        np.save("data/test/test_labels", test_labels)

        print("Data processing completed.")

    except:
        print("Data processing failed.")
        exit(0)
