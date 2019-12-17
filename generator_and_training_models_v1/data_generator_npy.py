# 4 classes
# image size 28*28*1
# pixel value [0,1]

##### set specific gpu #####
import os
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"]="1"

import tensorflow as tf

from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import numpy as np
import random
import time
import shutil
import pandas as pd
# import csv
##### 
physical_devices = tf.config.experimental.list_physical_devices('GPU')
# assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
tf.config.experimental.set_memory_growth(physical_devices[0], True)
# random.random() Return the next random floating point number in the range [0.0, 1.0).


image_size = 28
divide_num = 14

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
                if i < image_size/divide_num and j >= image_size/divide_num:
                    image[i][j] = 1.
                else:
                    image[i][j] = random.random()
    elif label == 2:
        for i in range(image_size):
            for j in range(image_size):
                if i >= image_size/divide_num and j < image_size/divide_num:
                    image[i][j] = 1.
                else:
                    image[i][j] = random.random()
    elif label == 3:
        for i in range(image_size):
            for j in range(image_size):
                if i >= image_size/divide_num and j >= image_size/divide_num:
                    image[i][j] = 1.
                else:
                    image[i][j] = random.random()
    else:
        assert False, "this class does not exist"

    return image

# generated_image = []
# for i in range(4):
#     generated_image.append(generate_base_image(i))
# generated_image = np.array(generated_image)
# print(generated_image.shape)
# generated_image = generated_image.reshape((4,image_size,image_size,1))
# print(generated_image.shape)
# plt.imshow(generated_image,cmap="gray")
# plt.show()
# time.sleep(10)

def image_transfer_model():
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same',kernel_initializer='random_uniform',bias_initializer='random_normal',input_shape=(image_size, image_size,1)))
    model.add(layers.BatchNormalization())
        # model.add(layers.dropout())
    # model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same',kernel_initializer='random_uniform',bias_initializer='random_normal'))
    # model.add(layers.BatchNormalization())
    # # model.add(layers.dropout())
    # model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same',kernel_initializer='random_uniform',bias_initializer='random_normal'))
    # model.add(layers.BatchNormalization())
    # # # model.add(layers.dropout())
    # model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same',kernel_initializer='random_uniform',bias_initializer='random_normal'))
    # model.add(layers.BatchNormalization())
    # # # model.add(layers.dropout())
    # model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same',kernel_initializer='random_uniform',bias_initializer='random_normal'))
    # model.add(layers.BatchNormalization())
    # model.add(layers.Dropout(0.1))
    model.add(layers.Conv2D(1, (3, 3), activation='relu', padding='same',kernel_initializer='random_uniform',bias_initializer='random_normal'))
    return model

# generated_image = tf.convert_to_tensor(generated_image)
# generator = image_transfer_model()

# generated_image = generator(generated_image,training=False)
# print(generated_image)
# print(generated_image[0].shape)
# plt.imshow(generated_image[0, :, :, 0],cmap="gray")
# plt.show()
# plt.savefig("image{}.png".format(str(1)))
# time.sleep(1)
# plt.imshow(generated_image[1, :, :, 0],cmap="gray")
# plt.show()
# time.sleep(1)
# plt.imshow(generated_image[2, :, :, 0],cmap="gray")
# plt.show()
# time.sleep(1)
# plt.imshow(generated_image[3, :, :, 0],cmap="gray")
# plt.show()
# time.sleep(1)

image_dir = "datasets"
classes = [0,1,2,3]
def image_creator(image_num):
    generator = image_transfer_model()
    for a_class in classes:
        # savefig_path = os.path.join(image_dir,str(a_class))
        # if os.path.exists(savefig_path):
        #     shutil.rmtree(savefig_path)
        # os.makedirs(savefig_path)
        generated_image = []
        for num in range(image_num):
            generated_image.append(generate_base_image(a_class))
        generated_image = np.array(generated_image)
        generated_image = generated_image.reshape((image_num,image_size,image_size,1))
        generated_image = tf.convert_to_tensor(generated_image)
        generated_image = generator(generated_image,training=False)
        # print("type",type(generated_image))
        # print("data",generated_image[0])
        generated_image = np.array(generated_image)
        # print("type",type(generated_image))
        np.save("npy_datasets/"+str(a_class)+'.npy', generated_image)
        # pd.DataFrame(generated_image).to_csv("csv_datasets/"+str(a_class)+'.csv')
        # with open("csv_datasets/"+str(a_class)+'.csv','w') as csvfile:
        #     csv.writer(csvfile).writerows(generated_image)
        # generated_image = pd.DataFrame(generated_image)
        # print("after pandas transformation")
        # print("type",type(generated_image))
        # print("before sleep")
        # os.sleep(100)
        # print("after sleep")
        # for num in range(image_num):
        #     plt.imshow(generated_image[num, :, :, 0],cmap="gray")
        #     plt.savefig(savefig_path + "/image{}.png".format(str(num)))

image_creator(1000)
