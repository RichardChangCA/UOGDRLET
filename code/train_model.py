import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import time
from tensorflow.keras.utils import to_categorical
from tensorflow.python.framework import graph_util
tf.reset_default_graph()
batch_size = 16
epochs = 3
lr_decay_epoch  = 3
image_batch = 1000
batch_amount = 1
image_amount = batch_amount*image_batch
# image amount should be multiplication of image_batch and batch_amount

# model_complexity is the complexity of training model
complexity_level = 4
num_layers = 18
datasets_name = "npy_datasets-%d"%complexity_level
image_size = 28
num_classes = 4
classes = [0,1,2,3]

# complexity_level is the complexity of dataset
model_dir_name = "model_with_complexity_" + str(complexity_level) + "_dataset_complexity_" + str(complexity_level)

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

x_train, x_test, y_train, y_test = train_test_split(images_sum, labels_sum, test_size=0.1, random_state=50)
a= x_train[:,:,:,0]
b = y_train

n_sample = x_train.shape[0]
n_batch = int((n_sample+batch_size-1) / batch_size)
# print(train_images.shape)

### check image
image_num = 99
# =============================================================================
# print(x_train[image_num,:,:,0])
# =============================================================================
plt.imshow(x_train[image_num,:,:,0])
plt.colorbar()
plt.show()
print(y_train[image_num])

y_train =to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)

input_x = tf.placeholder(tf.float32, [None, 28, 28, 1], name='input_x')
y = tf.placeholder(tf.uint8, [None, num_classes], name='input_y')
with tf.variable_scope('model'):
    x = tf.contrib.layers.conv2d(input_x, 32, 3,  padding='valid')
    x = tf.contrib.layers.conv2d(x, 64, 3,  padding='valid')
    x = tf.contrib.layers.conv2d(x, 96, 3,  padding='valid')
    x = tf.contrib.layers.conv2d(x, 128, 3,  padding='valid')
    x = tf.contrib.layers.conv2d(x, 128, 3,  padding='valid')
    x = tf.contrib.layers.conv2d(x, 256, 3,  padding='same')
    x = tf.contrib.layers.conv2d(x, 256, 3,  padding='same')
    x = tf.contrib.layers.conv2d(x, 512, 3,  padding='same')
    x = tf.contrib.layers.conv2d(x, 512, 3,  padding='same')
    x = tf.contrib.layers.conv2d(x, 512, 3,  padding='same')
    x = tf.contrib.layers.conv2d(x, 512, 3,  padding='same')
    x = tf.layers.flatten(x)
    x = tf.layers.dense(x, 256, activation='relu')
    x = tf.layers.dense(x, 256, activation='relu')
    x = tf.layers.dense(x, 128, activation='relu')
    x = tf.layers.dense(x, 128, activation='relu')
    x = tf.layers.dense(x, 128, activation='relu')
    x = tf.layers.dense(x, 128, activation='relu')
    x = tf.layers.dense(x, 128, activation='relu')
    x = tf.layers.dense(x, 128, activation='relu')
    x = tf.layers.dense(x, 32, activation='relu')
    logits = tf.layers.dense(x, num_classes, name='logits')
# =============================================================================
#     logits = tf.reduce_mean(x, axis=[1,2])
# =============================================================================
# =============================================================================
#     x = tf.layers.flatten(input_x)
#     x = tf.layers.dense(x, 64, activation='relu', name='fc1')
#     x = tf.layers.dense(x, 64, activation='relu', name='fc2')
#     x = tf.layers.dense(x, 64, activation='relu', name='fc3')
#     logits = tf.layers.dense(x, 4, activation=None, name='fc4')
# =============================================================================
    print(logits.get_shape())
with tf.variable_scope('acc'):
    count = tf.equal(tf.argmax(y, axis=1), tf.argmax(logits, axis=1))
    acc = tf.reduce_mean(tf.cast(count, tf.float32), name='acc')
with tf.variable_scope('loss'):
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=logits))
with tf.variable_scope('train_op'):
    global_step = tf.Variable(0,trainable=False)
    learning_rate = tf.train.exponential_decay(0.00005,global_step,lr_decay_epoch*n_batch,0.5,staircase=True)
    train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step)
init = tf.global_variables_initializer()

# =============================================================================
# p=0
# for i in [n.name for n in tf.get_default_graph().as_graph_def().node]:
#     print(i)
#     p=p+1
#     if p ==100:
#         break
# =============================================================================
    
def evaluate(x_test, y_test, batch_size=100):
    print('\nEvaluating')
    n_sample = x_test.shape[0]
    n_batch = int((n_sample+batch_size-1) / batch_size)
    avg_loss, avg_acc = 0, 0
    for batch in range(n_batch):
        print(' batch {0}/{1}'.format(batch + 1, n_batch), end='\r')
        start = batch * batch_size
        end = min(n_sample, start + batch_size)
        cnt = end - start
        batch_loss, batch_acc = sess.run(
            [loss, acc],
            feed_dict={input_x: x_test[start:end],
                       y: y_test[start:end]})
        avg_loss += batch_loss * cnt
        avg_acc += batch_acc * cnt
    avg_loss /= n_sample
    avg_acc /= n_sample
    print('loss: {0:.4f} acc: {1:.4f}'.format(avg_loss, avg_acc))
    return avg_loss, avg_acc


with tf.Session() as sess:
    sess.run(init)
    for epoch in range(epochs):
        print('\nEpoch {0}/{1} training...'.format(epoch + 1, epochs))
        for batch in range(n_batch):
            print(' batch {0}/{1}'.format(batch + 1, n_batch), end='\r')
            start = batch * batch_size
            end = min(n_sample, start + batch_size)
            _, lr = sess.run([train_op,learning_rate], feed_dict={input_x: x_train[start:end],
                                          y: y_train[start:end]})
        _, accuracy = evaluate(x_test, y_test)
        print(lr)
    constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph_def, ["acc/acc"])
    with tf.gfile.FastGFile('./model/model-c%d-l%d-acc%f.pb'%(complexity_level,num_layers,accuracy), mode='wb') as f:
        f.write(constant_graph.SerializeToString())
