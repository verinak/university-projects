import tensorflow as tf
import cv2  # byet3amel m3 el sowar
import numpy as np
import matplotlib.pyplot as plt  # visualization

# load data to get the test data and then normalize it
mnist = tf.keras.datasets.mnist
(x_train, y_train) , (x_test, y_test) = mnist.load_data()
x_test = tf.keras.utils.normalize(x_test, axis = 1)

# load the model we saveed in train.py
model = tf.keras.models.load_model('handwritten.model')

# test the model using the testing data and print the loss and accuracy
loss, accuracy = model.evaluate(x_test, y_test)

print(loss)
print(accuracy)



## testing on my own handwriting on 28x28 image
## uncomment and fix path to run code

img = cv2.imread('D:\\Downloads\\digits\\test_3.png')[:,:,0]
img = np.invert(np.array([img]))
# invert picture array to get black on white image

# print prediction and show image with pyplot
prediction = model.predict(img)
print('Prediction: ', np.argmax(prediction))
plt.imshow(img[0], cmap=plt.cm.binary)
plt.show()