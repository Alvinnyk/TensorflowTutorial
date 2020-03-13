import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

'''
    ========== Tutorial 2 ==========
    Simple sequential neural network to classify different classes of clothing
    
    Fashion dataset from MNIST contains 70000 (28 x 28) images of 10 different classes of clothing.
    Example bitmap image: Ankle Boot
    [[  0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0]
     [  0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0]
     [  0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0]
     [  0   0   0   0   0   0   0   0   0   0   0   0   1   0   0  13  73   0   0   1   4   0   0   0   0   1   1   0]
     [  0   0   0   0   0   0   0   0   0   0   0   0   3   0  36 136 127  62  54   0   0   0   1   3   4   0   0   3]
     [  0   0   0   0   0   0   0   0   0   0   0   0   6   0 102 204 176 134 144 123  23   0   0   0   0  12  10   0]
     [  0   0   0   0   0   0   0   0   0   0   0   0   0   0 155 236 207 178 107 156 161 109  64  23  77 130  72  15]
     [  0   0   0   0   0   0   0   0   0   0   0   1   0  69 207 223 218 216 216 163 127 121 122 146 141  88 172  66]
     [  0   0   0   0   0   0   0   0   0   1   1   1   0 200 232 232 233 229 223 223 215 213 164 127 123 196 229   0]
     [  0   0   0   0   0   0   0   0   0   0   0   0   0 183 225 216 223 228 235 227 224 222 224 221 223 245 173   0]
     [  0   0   0   0   0   0   0   0   0   0   0   0   0 193 228 218 213 198 180 212 210 211 213 223 220 243 202   0]
     [  0   0   0   0   0   0   0   0   0   1   3   0  12 219 220 212 218 192 169 227 208 218 224 212 226 197 209  52]
     [  0   0   0   0   0   0   0   0   0   0   6   0  99 244 222 220 218 203 198 221 215 213 222 220 245 119 167  56]
     [  0   0   0   0   0   0   0   0   0   4   0   0  55 236 228 230 228 240 232 213 218 223 234 217 217 209  92   0]
     [  0   0   1   4   6   7   2   0   0   0   0   0 237 226 217 223 222 219 222 221 216 223 229 215 218 255  77   0]
     [  0   3   0   0   0   0   0   0   0  62 145 204 228 207 213 221 218 208 211 218 224 223 219 215 224 244 159   0]
     [  0   0   0   0  18  44  82 107 189 228 220 222 217 226 200 205 211 230 224 234 176 188 250 248 233 238 215   0]
     [  0  57 187 208 224 221 224 208 204 214 208 209 200 159 245 193 206 223 255 255 221 234 221 211 220 232 246   0]
     [  3 202 228 224 221 211 211 214 205 205 205 220 240  80 150 255 229 221 188 154 191 210 204 209 222 228 225   0]
     [ 98 233 198 210 222 229 229 234 249 220 194 215 217 241  65  73 106 117 168 219 221 215 217 223 223 224 229  29]
     [ 75 204 212 204 193 205 211 225 216 185 197 206 198 213 240 195 227 245 239 223 218 212 209 222 220 221 230  67]
     [ 48 203 183 194 213 197 185 190 194 192 202 214 219 221 220 236 225 216 199 206 186 181 177 172 181 205 206 115]
     [  0 122 219 193 179 171 183 196 204 210 213 207 211 210 200 196 194 191 195 191 198 192 176 156 167 177 210  92]
     [  0   0  74 189 212 191 175 172 175 181 185 188 189 188 193 198 204 209 210 210 211 188 188 194 192 216 170   0]
     [  2   0   0   0  66 200 222 237 239 242 246 243 244 221 220 193 191 179 182 182 181 176 166 168  99  58   0   0]
     [  0   0   0   0   0   0   0  40  61  44  72  41  35   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0]
     [  0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0]
     [  0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0]]
'''

if __name__ == "__main__":

    # set to True if you want to retrain model
    retrain_model = False

    train_images = []
    train_labels = []
    test_images = []
    test_labels = []

    # names of the different classes of input images
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

    try:
        print("Loading dataset..")
        train_images = np.load("data/train/train_images.npy")
        train_labels = np.load("data/train/train_labels.npy")
        test_images = np.load("data/test/test_images.npy")
        test_labels = np.load("data/test/test_labels.npy")
    except:
        print("Failed to load dataset")
        exit(0)

    # processing images so each cell value is [0:1]
    train_images = train_images / 255.0
    test_images = test_images / 255.0

    # Define model of neural network
    '''
        relu: Rectified Linear Unit
                A(x) = max(0,x)
        sigmoid: Sigmoid Function
                A(x) = 1/(1 + e^(-x))
        softmax: Softmax Function
                Similar to sigmoid
                it returns the probabilities of each class and the target class will have the high probability
    '''
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10, activation="softmax")
    ])

    # Compile the model by choosing the optimizer, loss_function, and metrics
    '''
        logits: The vector of raw (non-normalized) predictions that a classification model generates
        The model's last layer has "softmax" activation, hence it is not the raw output of the model.
    '''
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                  metrics=['accuracy'])

    print(model.summary())

    try:
        if retrain_model:
            raise Exception()
        model.load_weights("model/model")
    except:
        model.fit(train_images, train_labels, epochs=10)
        test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
        print('\nTest accuracy:', test_acc)
        model.save_weights("model/model")

    all_images = np.concatenate((train_images, test_images))
    all_labels = np.concatenate((train_labels, test_labels))

    print("============================================================")
    print("     Select an index [0:{}] to see prediction".format(len(all_images)))
    print("============================================================")
    while True:
        i = input("index: ")
        if not i.isdigit():
            print("Invalid input")
            continue
        i = int(i)
        if i >= len(all_images):
            print("Out of bounds")
            continue

        # get the image
        image = all_images[i]
        image_array = np.array([all_images[i]])

        # prediction from model
        prediction_array = model.predict(image_array)
        prediction = prediction_array[0]

        # true label for image
        predicted_label = np.argmax(prediction)
        true_label = all_labels[i]

        '''
            matplotlib functions to plot diagram
        '''
        plt.figure(figsize=(6, 3))

        # plot image
        plt.subplot(1, 2, 1)
        plt.grid(False)
        plt.xticks([])
        plt.yticks([])
        plt.imshow(image, cmap=plt.cm.binary)
        if predicted_label == true_label:
            color = 'blue'
        else:
            color = 'red'
        plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                             100 * np.max(prediction),
                                             class_names[true_label]),
                   color=color)

        # plot value array
        plt.subplot(1, 2, 2)
        plt.grid(False)
        plt.xticks(range(10))
        plt.yticks([])
        thisplot = plt.bar(range(10), prediction, color="#777777")
        plt.ylim([0, 1])
        thisplot[predicted_label].set_color('red')
        thisplot[true_label].set_color('blue')

        plt.show()
