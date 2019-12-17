import tensorflow as tf
import numpy as np
import os
import pathlib
import matplotlib.pyplot as plt
from tensorflow.keras import datasets, layers, models
from sklearn.model_selection import train_test_split
import datetime
import time
import shutil

physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

image_batch = 1000
batch_amount = 10
image_amount = batch_amount*image_batch
# image amount should be multiplication of image_batch and batch_amount
datasets_name = "npy_datasets"
image_size = 28
classes = [0,1,2,3]
model_complexity = 10
# model_complexity is the complexity of training model
complexity_level = 6
# complexity_level is the complexity of dataset
model_dir_name = "model_with_complexity_" + str(model_complexity) + "_dataset_complexity_" + str(complexity_level)

# load datasets
for a_class in classes:
    for n in range(batch_amount):
        images = np.load(datasets_name+"/"+str(a_class)+'/'+ str(n) +'.npy')
        images = images.reshape((image_batch,image_size,image_size,1))
        labels = np.zeros(image_batch)+a_class
        if a_class == 0 and n == 0:
            images_sum = images
            labels_sum = labels
        else:
            images_sum = np.concatenate((images_sum, images), axis=0)
            labels_sum = np.concatenate((labels_sum, labels), axis=0)

train_images, test_images, train_labels, test_labels = train_test_split(images_sum, labels_sum, test_size=0.3, random_state=50)

# print(train_images.shape)

### check image
# image_num = 0
# print(train_images[image_num,:,:,0])
# plt.imshow(train_images[image_num,:,:,0],cmap="gray")
# plt.show()
# time.sleep(1)

model = models.Sequential()
model.add(layers.Conv2D(16, (5, 5),padding='same', input_shape=(28, 28, 1)))
model.add(layers.ReLU())
model.add(layers.BatchNormalization())
for _ in range(model_complexity):
    model.add(layers.Conv2D(32, (3, 3), padding='same'))
    model.add(layers.ReLU())
    model.add(layers.BatchNormalization())
    # model.add(layers.Dropout(0.3))
model.add(layers.Flatten())
# model.add(layers.MaxPooling2D(image_size,image_size))
model.add(layers.Dense(4, activation='softmax'))
# model.add(layers.Conv2D(4, (1,1),activation='softmax'))
model.summary()

# Pooling do not work well
# Dropout do not work well

log_dir="logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(train_images, train_labels, batch_size=256, epochs=50, 
                    validation_data=(test_images, test_labels), callbacks=[tensorboard_callback])

# print("history:",history)

if os.path.exists(model_dir_name):
    shutil.rmtree(model_dir_name)
os.mkdir(model_dir_name)

tf.saved_model.save(model, model_dir_name + "/")
#save model variables and other information in .pb file
