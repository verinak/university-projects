### Name: Verina Michel Asham Abdel Malak - فيرينا ميشيل عشم عبد الملاك
### ID: 20221440977
### Introduction to Artificial Intelligence Project

### Implementation of a neural network model to recognize handwritten digits using tenserflow ###

import tensorflow as tf

# get built-in handwritten digits database
mnist = tf.keras.datasets.mnist

# load training and testing data
(x_train, y_train) , (x_test, y_test) = mnist.load_data()
# x_train and x_test represent the training and testing image pixels
# y_train and y_test represent the training and testing labels/classification

# normalizing the training data to improve the accuracy
x_train = tf.keras.utils.normalize(x_train, axis = 1)

# Neural Network linear model
model = tf.keras.models.Sequential()


# adding the neural network layers
# first the input layer to flatten input into 1d lisr
# the input shape is 28 x 28 pixels
model.add(tf.keras.layers.Flatten(input_shape = (28,28)))
# adding 2 hidden layers with 128 neurons that use rl Rectified Linear Unit activation function
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dense(128, activation='relu'))
# adding output layer with 10 neurons for the 10 digits (0-9) with Softmax activation function
model.add(tf.keras.layers.Dense(10, activation='softmax'))   # output layer?

# compiling the model
# Optimizer is the algorithms used to change the weights and learning rate to reduce the error
# Loss function is used to calculate error
# We will use the accracy metric to evaluate our model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# training the model
model.fit(x_train, y_train, epochs=5)

# save the model after training to use in test.py
model.save('handwritten.model')