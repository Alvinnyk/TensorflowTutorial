import tensorflow as tf
import numpy as np

'''
    ========== Tutorial 4 ==========
    Comparing CNN and SNN for image classification of digits
    Both trained on the same dataset
    
    =============== SNN ===============                 =============== CNN ===============
    Model:      128d -> 128d -> 10d                     Model:      64c2p -> 64c2p -> 10d
    t_params:   118,282                                 t_params:   68,938
    val_acc:    ~ 0.97                                  val_acc:    ~ 0.99
    
    CNN has overall better performace over SNN.
    CNN manages a better validation accuracy with about half as many parameters as SNN.
    
    Adding a dense layer between the last Conv2D layer and the output layer seems to hurt performance.
    MaxPooling on every Conv2D layer increases performance over non MaxPooling models
    A Conv2D layer of 64 filters seems to be the sweet spot. Bumping to 128 filters will hurt performance.
'''

def SNN_model(img_data, img_label):

    # Tensorflow neural network model
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=img_data[0].shape),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    # Compile the model by choosing the optimizer, loss_function, and metrics
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                  metrics=['accuracy'])

    print(model.summary())
    print("==================================================")
    print("     SNN Model: 128d-128d-10d")
    print("==================================================")

    # Trains, evaluates and saves the model
    model.fit(img_data, img_label, validation_split=0.2, epochs=7, verbose=2)

    model.save("model/SNN_model.h5")

    return model


def CNN_model(img_data, img_label):

    img_data_reshaped = np.array(img_data).reshape((-1, 28, 28, 1))

    model = tf.keras.Sequential([
        # input layer with img_data.shape
        tf.keras.layers.Input(shape=img_data_reshaped[0].shape),

        # First Conv2D layer
        tf.keras.layers.Conv2D(64, kernel_size=(3, 3), padding="same", activation="relu"),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2, padding="valid"),

        # second Conv2D layer
        tf.keras.layers.Conv2D(64, kernel_size=(3, 3), padding="same", activation="relu"),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2, padding="valid"),

        # flatten out the 2D layer and feed into a dense layer of 32 neurons
        tf.keras.layers.Flatten(),
        # tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(10, activation="softmax")
    ])

    model.compile(optimizer="adam",
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                  metrics=["accuracy"])
    print(model.summary())
    print("==================================================")
    print("     CNN Model: 64c2p-64c2p-10d")
    print("==================================================")

    # Trains, evaluates and saves the model
    model.fit(img_data_reshaped, img_label, validation_split=0.2, epochs=7, verbose=2)

    model.save("model/CNN_model.h5")

    return model

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

    model_SNN = SNN_model(img_data, img_label)
    model_CNN = CNN_model(img_data, img_label)



