# 4 classes
# image size 28*28*1
# pixel value [0,1]

##### set specific gpu #####
import os
import tensorflow as tf
from PIL import Image
import numpy as np
import random
import shutil
from matplotlib import pyplot as plt

tf.reset_default_graph()
image_size = 28
divide_num = 4
classes = [0,1,2,3]
image_batch = 1000
batch_amount = 1
image_amount = batch_amount*image_batch
complexity_level = 5
# image amount should be multiplication of image_batch and batch_amount
datasets_name = "npy_datasets-%d"%complexity_level

def plot_demo(complexity_level):
    demo_image1 = processed_image_list[0][0,:,:]
    demo_image2 = processed_image_list[1][0,:,:]
    demo_image3 = processed_image_list[2][0,:,:]
    demo_image4 = processed_image_list[3][0,:,:]
    plt.figure(1, figsize=(12,3))
    plt.suptitle("data complexity = %d"%complexity_level)
    plt.gcf().subplots_adjust(right=0.85)
    plt.subplot(141)
    plt.title("class 0")
    plt.imshow(demo_image1)
    plt.subplot(142)
    plt.title("class 1")
    plt.imshow(demo_image2)
    plt.subplot(143)
    plt.title("class 2")
    plt.imshow(demo_image3)
    plt.subplot(144)
    plt.title("class 3")
    plt.imshow(demo_image4)
    cbar_ax = plt.gcf().add_axes([0.9, 0.1, 0.05, 0.8])
    plt.colorbar(cax=cbar_ax)
    plt.savefig(datasets_name+"/demo-%d.png"%complexity_level)
    plt.show()
    return demo_image1

def generate_base_image(label, image_size, complexity_level):
    image_size = image_size + 2 * complexity_level
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
    return image*1000


#%%  Model

input_x = tf.placeholder(tf.float32, [None, None, None, 1])
x = tf.contrib.layers.conv2d(input_x, 32, 3, padding='valid', activation_fn=tf.nn.leaky_relu, biases_initializer=tf.random_normal_initializer())
for _ in range(complexity_level-2):
    x = tf.contrib.layers.conv2d(x, 32, 3, padding='valid', activation_fn=tf.nn.leaky_relu, biases_initializer=tf.random_normal_initializer())
x = tf.contrib.layers.conv2d(x, 1, 3, padding='valid', activation_fn=tf.nn.leaky_relu, biases_initializer=tf.random_normal_initializer())
x = tf.squeeze(x)
x = tf.image.per_image_standardization(x)

#%%
def image_creator(processed_image_list, image_num,complexity_level):
    if os.path.exists(datasets_name):
        shutil.rmtree(datasets_name)
    os.mkdir(datasets_name)
    with tf.Session() as sess:  
        sess.run(tf.global_variables_initializer())
        for a_class in classes:
            print("class:%d is being processed.."%a_class)
            os.mkdir(datasets_name+'/'+str(a_class))
            for batch in range(int(image_num/image_batch)):
                generated_image = []
                for num in range(image_batch):
                    generated_image.append(generate_base_image(a_class, image_size, complexity_level))
                generated_image = np.array(generated_image, np.float32)
                generated_image = generated_image.reshape((image_batch,image_size+2*complexity_level,image_size+2*complexity_level,1))
                if complexity_level == 0:
                    np.save(datasets_name+'/'+str(a_class)+'/'+ str(batch) +'.npy', generated_image)
                    processed_image_list.append(generated_image)
                    break
                processed_image = sess.run(x, feed_dict={input_x:generated_image})
                np.save(datasets_name+'/'+str(a_class)+'/'+ str(batch) +'.npy', processed_image)
                processed_image_list.append(processed_image)


processed_image_list = []
image_creator(processed_image_list, image_amount,complexity_level)
demo_image1 = plot_demo(complexity_level)
