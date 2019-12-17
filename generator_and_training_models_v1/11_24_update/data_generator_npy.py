# 4 classes
# image size 28*28*1
# pixel value [0,1]

##### set specific gpu #####
import os
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import numpy as np
import random
import time
import shutil
import pandas as pd

##### 
physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)


image_size = 28
divide_num = 2
image_dir = "datasets"
classes = [0,1,2,3]
image_amount = 2000
complexity_level = 4

def generate_base_image(label):
    image = np.zeros((image_size,image_size),dtype=np.float64)
    if label == 0:
        for i in range(image_size):
            for j in range(image_size):
                if i < image_size/divide_num and j < image_size/divide_num:
                    image[i][j] = 1.
                else:
                    image[i][j] = random.random()
    elif label == 1:
        for i in range(image_size):
            for j in range(image_size):
                if i < image_size/divide_num and j >= image_size - image_size/divide_num:
                    image[i][j] = 1.
                else:
                    image[i][j] = random.random()
    elif label == 2:
        for i in range(image_size):
            for j in range(image_size):
                if i >= image_size - image_size/divide_num and j < image_size/divide_num:
                    image[i][j] = 1.
                else:
                    image[i][j] = random.random()
    elif label == 3:
        for i in range(image_size):
            for j in range(image_size):
                if i >= image_size - image_size/divide_num and j >= image_size - image_size/divide_num:
                    image[i][j] = 1.
                else:
                    image[i][j] = random.random()
    else:
        assert False, "this class does not exist"

    return image


def image_transfer_model(complexity_level):
    model = models.Sequential()
    model.add(layers.Conv2D(16, (3, 3), activation='relu', padding='same',kernel_initializer='random_uniform',bias_initializer='random_normal',input_shape=(image_size, image_size,1)))
    model.add(layers.BatchNormalization())
    for _ in range(complexity_level):
        model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same',kernel_initializer='random_uniform',bias_initializer='random_normal',input_shape=(image_size, image_size,1)))
        model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(1, (3, 3), activation='relu', padding='same',kernel_initializer='random_uniform',bias_initializer='random_normal'))
    return model



def image_creator(image_num,complexity_level):
    generator = image_transfer_model(complexity_level)
    for a_class in classes:
        generated_image = []
        for num in range(image_num):
            generated_image.append(generate_base_image(a_class))
        generated_image = np.array(generated_image)
        generated_image = generated_image.reshape((image_num,image_size,image_size,1))
        generated_image = tf.convert_to_tensor(generated_image)
        generated_image = generator(generated_image,training=False)
        generated_image = np.array(generated_image)
        np.save("npy_datasets/"+str(a_class)+'.npy', generated_image)

image_creator(image_amount,complexity_level)
