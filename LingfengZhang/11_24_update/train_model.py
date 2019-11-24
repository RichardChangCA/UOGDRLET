import tensorflow as tf
import numpy as np
import os
import pathlib
import matplotlib.pyplot as plt
from tensorflow.keras import datasets, layers, models
from sklearn.model_selection import train_test_split
import datetime

physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

image_num = 2000
image_size = 28
classes = [0,1,2,3]
model_complexity = 5

for a_class in classes:
    images = np.load("npy_datasets/"+str(a_class)+'.npy')
    images = images.reshape((image_num,image_size,image_size,1))
    labels = np.zeros(image_num)+a_class
    if a_class == 0:
        images_sum = images
        labels_sum = labels
    else:
        images_sum = np.concatenate((images_sum, images), axis=0)
        labels_sum = np.concatenate((labels_sum, labels), axis=0)

train_images, test_images, train_labels, test_labels = train_test_split(images_sum, labels_sum, test_size=0.3, random_state=50)

model = models.Sequential()
model.add(layers.Conv2D(16, (5, 5), activation='relu',padding='same', input_shape=(28, 28, 1)))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D(2,2))

for _ in range(model_complexity):
    model.add(layers.Conv2D(32, (3, 3), activation='relu',padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.3))

model.add(layers.Flatten())
# model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(4, activation='softmax'))
model.summary()

log_dir="logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(train_images, train_labels, epochs=1000, 
                    validation_data=(test_images, test_labels), callbacks=[tensorboard_callback])
