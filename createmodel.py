# Model based on: https://github.com/animikhaich/ECG-Atrial-Fibrillation-Classification-Using-CNN
import tensorflow as tf
# import keras
# from keras.models import Sequential
# from keras.layers import Input, Dense, Conv1D, Dropout, MaxPool1D, Flatten
# from keras import backend as K

def create_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv1D(filters=512, kernel_size=32, padding='same', kernel_initializer='normal', activation=tf.nn.relu, input_shape=(1280, 2)),
        # tf.keras.layers.Dense(4096,kernel_initializer='normal', activation=tf.nn.relu),
        tf.keras.layers.Conv1D(filters=512, kernel_size=32, padding='same', kernel_initializer='normal', activation=tf.nn.relu),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Conv1D(filters=256, kernel_size=32, padding='same', kernel_initializer='normal', activation=tf.nn.relu),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Conv1D(filters=256, kernel_size=32, padding='same', kernel_initializer='normal', activation=tf.nn.relu),
        tf.keras.layers.MaxPool1D(pool_size=128),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(units=128, kernel_initializer='normal', activation=tf.nn.relu),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(1,kernel_initializer='normal', activation=tf.nn.softmax)
    ])

    # optimizer = tf.keras.optimizers.SGD(lr=0.001, momentum=0.5)
    # model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss=tf.keras.losses.sparse_categorical_crossentropy,
                  metrics=['accuracy'])
    
    return model



# Build the model

    # # The model architecture type is sequential hence that is used
    # model = Sequential()

    # # We are using 4 convolution layers for feature extraction
    # model.add(Conv1D(filters=512, kernel_size=32, padding='same', kernel_initializer='normal', activation=tf.nn.relu, input_shape=(1280, 2)))
    # model.add(Conv1D(filters=512, kernel_size=32, padding='same', kernel_initializer='normal', activation=tf.nn.relu))
    # model.add(Dropout(0.2)) # This is the dropout layer. It's main function is to inactivate 20% of neurons in order to prevent overfitting
    # model.add(Conv1D(filters=256, kernel_size=32, padding='same', kernel_initializer='normal', activation=tf.nn.relu))
    # model.add(Dropout(0.2))
    # model.add(Conv1D(filters=256, kernel_size=32, padding='same', kernel_initializer='normal', activation=tf.nn.relu))
    # model.add(MaxPool1D(pool_size=128)) # We use MaxPooling with a filter size of 128. This also contributes to generalization
    # model.add(Dropout(0.2))

    # # The prevous step gices an output of multi dimentional data, which cannot be fead directly into the feed forward neural network. Hence, the model is flattened
    # model.add(Flatten())
    # # One hidden layer of 128 neurons have been used in order to have better classification results
    # model.add(Dense(units=128, kernel_initializer='normal', activation=tf.nn.relu))
    # model.add(Dropout(0.5))
    # # The final neuron HAS to be 1 in number and cannot be more than that. This is because this is a binary classification problem and only 1 neuron is enough to denote the class '1' or '0'
    # model.add(Dense(units=1, activation='sigmoid'))

    # # Print the summary of the model
    # model.summary()
    # optimizer = keras.optimizers.SGD(lr=0.001, momentum=0.5)
    # model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])