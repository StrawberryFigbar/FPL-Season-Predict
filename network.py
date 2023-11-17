import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from keras.utils import to_categorical
import os
from plotter import *
import pandas as pd

# Sets a batch size
BATCH_SIZE = 2
# Sets the size we chop off for our test group
TEST_SIZE = 100
# Retrieves the Data from the csv
DataURL = f'final_data.csv'
Data = pd.read_csv(DataURL)
# Splits data into test and training groups
data_test = Data.tail(TEST_SIZE)
data_train = Data.iloc[:-TEST_SIZE]
# Splits the targets into test and training groups, changing targets to categorical arrays
target_test = data_test.pop('total_points_next')
target_test = to_categorical(target_test, num_classes=275)
target_train = data_train.pop('total_points_next')
target_train = to_categorical(target_train, num_classes=275)

# Converts the data to tensor arrays
tf.convert_to_tensor(data_test)
tf.convert_to_tensor(data_train)
# Normalizes the data
normalizer = tf.keras.layers.Normalization(axis=-1)
normalizer.adapt(data_train)

# Sets up model for the neural network


def get_basic_model():
    model = tf.keras.Sequential([
        normalizer,
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(275)
    ])
# Compiles the model
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    return model


# Sets up the location for storing the weights
checkpoint_path = "training_1/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)
# Sets up callback for saving weights while training
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)
# Creates model
model = get_basic_model()
# Loads in the weights for the model
model.load_weights(checkpoint_path)
# Trains the model
model.fit(data_train, target_train, epochs=40,
          batch_size=BATCH_SIZE, callbacks=[cp_callback])
nameURL = f'name_data.csv'
nameData = pd.read_csv(nameURL)
name_test = nameData.tail(TEST_SIZE)
print(model.layers[0].get_weights())
probability_model = tf.keras.Sequential([
    model,
    tf.keras.layers.Softmax()
])
predictions = probability_model.predict(data_test)

i = 3
print(
    f'Player: {name_test.iloc[i].at["first_name"]}, {name_test.iloc[i].at["second_name"]}')
print(f'Predicted Value: {np.argmax(predictions[i])}')
print(f'Real Value: {np.argmax(target_test[i])}')
plot_value_array(i, predictions[i],  target_test)
plot_predict(predictions, target_test)
print(model.layers[0].weights)
