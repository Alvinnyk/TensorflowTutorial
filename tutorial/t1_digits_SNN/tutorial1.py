import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

'''
    ========== Tutorial 1 ==========
    Simple sequential neural network example to classify handwritten digits

    Handwritten digit dataset from MNIST
    The data consists of 28 x 28 bitmap of a digit with each cell a value from 0:255
        len(train_data): 60000
        len(test_data):  10000
    Example:
        train_label[0]: 5 
        train_data[0]: 
    [[  0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0]
     [  0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0]
     [  0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0]
     [  0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0]
     [  0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0]
     [  0   0   0   0   0   0   0   0   0   0   0   0   3  18  18  18 126 136 175  26 166 255 247 127   0   0   0   0]
     [  0   0   0   0   0   0   0   0  30  36  94 154 170 253 253 253 253 253 225 172 253 242 195  64   0   0   0   0]
     [  0   0   0   0   0   0   0  49 238 253 253 253 253 253 253 253 253 251  93  82  82  56  39   0   0   0   0   0]
     [  0   0   0   0   0   0   0  18 219 253 253 253 253 253 198 182 247 241   0   0   0   0   0   0   0   0   0   0]
     [  0   0   0   0   0   0   0   0  80 156 107 253 253 205  11   0  43 154   0   0   0   0   0   0   0   0   0   0]
     [  0   0   0   0   0   0   0   0   0  14   1 154 253  90   0   0   0   0   0   0   0   0   0   0   0   0   0   0]
     [  0   0   0   0   0   0   0   0   0   0   0 139 253 190   2   0   0   0   0   0   0   0   0   0   0   0   0   0]
     [  0   0   0   0   0   0   0   0   0   0   0  11 190 253  70   0   0   0   0   0   0   0   0   0   0   0   0   0]
     [  0   0   0   0   0   0   0   0   0   0   0   0  35 241 225 160 108   1   0   0   0   0   0   0   0   0   0   0]
     [  0   0   0   0   0   0   0   0   0   0   0   0   0  81 240 253 253 119  25   0   0   0   0   0   0   0   0   0]
     [  0   0   0   0   0   0   0   0   0   0   0   0   0   0  45 186 253 253 150  27   0   0   0   0   0   0   0   0]
     [  0   0   0   0   0   0   0   0   0   0   0   0   0   0   0  16  93 252 253 187   0   0   0   0   0   0   0   0]
     [  0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0 249 253 249  64   0   0   0   0   0   0   0]
     [  0   0   0   0   0   0   0   0   0   0   0   0   0   0  46 130 183 253 253 207   2   0   0   0   0   0   0   0]
     [  0   0   0   0   0   0   0   0   0   0   0   0  39 148 229 253 253 253 250 182   0   0   0   0   0   0   0   0]
     [  0   0   0   0   0   0   0   0   0   0  24 114 221 253 253 253 253 201  78   0   0   0   0   0   0   0   0   0]
     [  0   0   0   0   0   0   0   0  23  66 213 253 253 253 253 198  81   2   0   0   0   0   0   0   0   0   0   0]
     [  0   0   0   0   0   0  18 171 219 253 253 253 253 195  80   9   0   0   0   0   0   0   0   0   0   0   0   0]
     [  0   0   0   0  55 172 226 253 253 253 253 244 133  11   0   0   0   0   0   0   0   0   0   0   0   0   0   0]
     [  0   0   0   0 136 253 253 253 212 135 132  16   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0]
     [  0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0]
     [  0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0]
     [  0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0]]
'''

if __name__ == "__main__":

    train_data = []
    train_label = []
    test_data = []
    test_label = []

    try:
        # Loads data stored locally
        print("Loading dataset...")
        train_data = np.load("data/train/train_data.npy")
        train_label = np.load("data/train/train_label.npy")
        test_data = np.load("data/test/test_data.npy")
        test_label = np.load("data/test/test_label.npy")
    except:
        print("Failed to load dataset.")
        exit(0)

    # the training and test_scripts data have cell values [0:255]
    # We now change them to [0, 1] for input to our neural network
    train_data = train_data / 255.0
    test_data = test_data / 255.0

    # Tensorflow neural network model
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10)
    ])

    # Compile the model by choosing the optimizer, loss_function, and metrics
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    # Attempts to load model from model folder, else it trains a new model
    try:
        # Loads the pre-trained model
        model.load_weights("model/model")
    except:
        # Trains, evaluates and saves the model
        model.fit(train_data, train_label, epochs=5)
        model.evaluate(test_data, test_label, verbose=2)

        # this saves the model as weights. We have to recreate and compile the same model to load the weights.
        model.save_weights("model/model")

    # combines all data and labels
    all_data = np.concatenate((train_data, test_data))
    all_label = np.concatenate((train_label, test_label))

    # matplotlib application to visualize predictions
    print("============================================================")
    print("     Select an index [0:{}] to see prediction".format(len(all_data)))
    print("============================================================")
    while True:
        i = input("Evaluate: ")
        if not i.isdigit():
            print("Invalid input")
            continue
        i = int(i)
        if i >= len(all_data):
            print("Out of bounds!")
            continue

        '''
            model.predict() method
                input:  a numpy array of data
                output: a numpy array of predictions
        '''
        data = np.array([all_data[int(i)]])     # frame input as an array of data -> an array of 1 data
        result = model.predict(data)            # get predictions from model
        result_index = np.argmax(result[0])     # read predictions from array of predictions -> first prediction

        print(result)
        print(result_index)

        plt.figure()
        plt.imshow(all_data[int(i)], cmap="gray_r")
        plt.colorbar()
        plt.grid(False)
        plt.title("data[{}] label: {} Prediction: {}".format(i, int(all_label[i]), result_index))
        plt.show()
