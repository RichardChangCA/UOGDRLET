import tensorflow as tf
import numpy as np
import os
import pathlib
import matplotlib.pyplot as plt
from tensorflow.keras import datasets, layers, models
from sklearn.model_selection import train_test_split
import datetime
# import csv

physical_devices = tf.config.experimental.list_physical_devices('GPU')
# assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
tf.config.experimental.set_memory_growth(physical_devices[0], True)

# data_dir = tf.keras.utils.get_file(fname='datasets')
# data_dir = pathlib.Path(data_dir)
# image_count = len(list(data_dir.glob('*/*.png')))
# print("image_count",image_count)

# CLASS_NAMES = np.array([item.name for item in data_dir.glob('*')])
# print("CLASS_NAMES",CLASS_NAMES)

# # The 1./255 is to convert from uint8 to float32 in range [0,1].
# image_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
# BATCH_SIZE = 32
# IMG_HEIGHT = 28
# IMG_WIDTH = 28
# STEPS_PER_EPOCH = np.ceil(image_count)
image_num = 1000
image_size = 28
classes = [0,1,2,3]
# classes = [0]
for a_class in classes:
    images = np.load("npy_datasets/"+str(a_class)+'.npy')
    # print("images",images)
    images = images.reshape((image_num,image_size,image_size,1))
    labels = np.zeros(image_num)+a_class
    # print("shape",images.shape)
    if a_class == 0:
        images_sum = images
        labels_sum = labels
    else:
        images_sum = np.concatenate((images_sum, images), axis=0)
        labels_sum = np.concatenate((labels_sum, labels), axis=0)

print(images_sum.shape)
print(labels_sum.shape)

train_images, test_images, train_labels, test_labels = train_test_split(images_sum, labels_sum, test_size=0.3, random_state=50)

# print(X_train.shape)
# BUFFER_SIZE = 10000
# dataset = tf.data.Dataset.from_tensor_slices((images_sum,labels_sum)).shuffle(BUFFER_SIZE)
# print(type(dataset))

# plt.imshow(images[0, :, :, 0],cmap="gray")
# plt.show()

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D((2, 2)))
# model.add(layers.Conv2D(64, (3, 3), activation='relu'))
# model.add(layers.MaxPooling2D((2, 2)))
# model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Flatten())
# model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(4, activation='softmax'))
model.summary()

log_dir="logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(train_images, train_labels, epochs=500, 
                    validation_data=(test_images, test_labels), callbacks=[tensorboard_callback])

# results = model.evaluate(test_images, test_labels, batch_size=64,callbacks=[tensorboard_callback])
# print("history",history)
# print("results",results)
