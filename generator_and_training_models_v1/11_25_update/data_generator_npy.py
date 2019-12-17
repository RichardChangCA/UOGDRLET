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
import datetime

##### 
physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)


image_size = 28
divide_num = 2
classes = [0,1,2,3]
image_batch = 1000
batch_amount = 10
image_amount = batch_amount*image_batch
# image amount should be multiplication of image_batch and batch_amount
complexity_level = 6
datasets_name = "npy_datasets"

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
    model.add(layers.Conv2D(16, (3, 3), padding='same',kernel_initializer='random_uniform',bias_initializer='random_normal',input_shape=(image_size, image_size,1)))
    model.add(layers.LeakyReLU(0.5))
    model.add(layers.BatchNormalization())
    for _ in range(complexity_level):
        model.add(layers.Conv2D(32, (3, 3), padding='same',kernel_initializer='random_uniform',bias_initializer='random_normal'))
        model.add(layers.LeakyReLU(0.5))
        model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(1, (3, 3), padding='same', kernel_initializer='random_uniform',bias_initializer='random_normal'))
    model.add(layers.LeakyReLU(0.5))
    return model



def image_creator(image_num,complexity_level):
    if os.path.exists(datasets_name):
        shutil.rmtree(datasets_name)
    os.mkdir(datasets_name)
    generator = image_transfer_model(complexity_level)
    for a_class in classes:
        os.mkdir(datasets_name+'/'+str(a_class))
        for batch in range(int(image_num/image_batch)):
            generated_image = []
            for num in range(image_batch):
                generated_image.append(generate_base_image(a_class))
            generated_image = np.array(generated_image)
            generated_image = generated_image.reshape((image_batch,image_size,image_size,1))
            generated_image = tf.convert_to_tensor(generated_image)
            generated_image = generator(generated_image,training=False)
            generated_image = np.array(generated_image)
            np.save(datasets_name+'/'+str(a_class)+'/'+ str(batch) +'.npy', generated_image)
            #release the memory of variable - generated_image
            del generated_image

image_creator(image_amount,complexity_level)
