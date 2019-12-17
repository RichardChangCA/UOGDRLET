
import os
import numpy as np

import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import tensorflow as tf
import sys
tf.reset_default_graph()
sys.path.append('../')
from attacks import fast_gradient


img_size = 28
img_chan = 1
n_classes = 10


print('\nLoading MNIST')

mnist = tf.keras.datasets.mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = np.reshape(X_train, [-1, img_size, img_size, img_chan])
X_train = X_train.astype(np.float32) / 255
X_test = np.reshape(X_test, [-1, img_size, img_size, img_chan])
X_test = X_test.astype(np.float32) / 255

to_categorical = tf.keras.utils.to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

print('\nSpliting data')

ind = np.random.permutation(X_train.shape[0])
X_train, y_train = X_train[ind], y_train[ind]

VALIDATION_SPLIT = 0.1
n = int(X_train.shape[0] * (1-VALIDATION_SPLIT))
X_valid = X_train[n:]
X_train = X_train[:n]
y_valid = y_train[n:]
y_train = y_train[:n]

print('\nConstruction graph')


def model(x, logits=False, training=False):
    with tf.variable_scope('conv0'):
        z = tf.layers.conv2d(x, filters=32, kernel_size=[3, 3],
                             padding='same', activation=tf.nn.relu)
        z = tf.layers.max_pooling2d(z, pool_size=[2, 2], strides=2)

    with tf.variable_scope('conv1'):
        z = tf.layers.conv2d(z, filters=64, kernel_size=[3, 3],
                             padding='same', activation=tf.nn.relu)
        z = tf.layers.max_pooling2d(z, pool_size=[2, 2], strides=2)

    with tf.variable_scope('flatten'):
        shape = z.get_shape().as_list()
        z = tf.reshape(z, [-1, np.prod(shape[1:])])

    with tf.variable_scope('mlp'):
        z = tf.layers.dense(z, units=128, activation=tf.nn.relu)
        z = tf.layers.dropout(z, rate=0.25, training=training)

    logits_ = tf.layers.dense(z, units=10, name='logits')
    y = tf.nn.softmax(logits_, name='ybar')

    if logits:
        return y, logits_
    return y



with tf.variable_scope('model'):
    x = tf.placeholder(tf.float32, (None, img_size, img_size, img_chan),
                           name='x')
    y = tf.placeholder(tf.float32, (None, n_classes), name='y')
    training = tf.placeholder_with_default(False, (), name='mode')

    ybar, logits = model(x, logits=True, training=training)

    with tf.variable_scope('acc'):
        count = tf.equal(tf.argmax(y, axis=1), tf.argmax(ybar, axis=1))
        acc = tf.reduce_mean(tf.cast(count, tf.float32), name='acc')

    with tf.variable_scope('loss'):
        xent = tf.nn.softmax_cross_entropy_with_logits(labels=y,
                                                       logits=logits)
        loss = tf.reduce_mean(xent, name='loss')

    with tf.variable_scope('train_op'):
        optimizer = tf.train.AdamOptimizer()
        train_op = optimizer.minimize(loss)

    saver = tf.train.Saver()

with tf.variable_scope('model', reuse=True):
    fgsm_eps = tf.placeholder(tf.float32, (), name='fgsm_eps')
    fgsm_epochs = tf.placeholder(tf.int32, (), name='fgsm_epochs')
    x_fgsm = fast_gradient.fgm(model, x, epochs=fgsm_epochs, eps=fgsm_eps)
    x_fgmt = fast_gradient.fgmt(model, x, epochs=fgsm_epochs, eps=fgsm_eps)

print('\nInitializing graph')

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
sess.run(tf.local_variables_initializer())


def evaluate(sess,  X_data, y_data, batch_size=128):
    print('\nEvaluating')

    n_sample = X_data.shape[0]
    n_batch = int((n_sample+batch_size-1) / batch_size)
    _loss, _acc = 0, 0

    for batch in range(n_batch):
        print(' batch {0}/{1}'.format(batch + 1, n_batch), end='\r')
        start = batch * batch_size
        end = min(n_sample, start + batch_size)
        cnt = end - start
        batch_loss, batch_acc = sess.run(
            [loss, acc],
            feed_dict={x: X_data[start:end],
                       y: y_data[start:end]})
        _loss += batch_loss * cnt
        _acc += batch_acc * cnt
    _loss /= n_sample
    _acc /= n_sample

    print(' loss: {0:.4f} acc: {1:.4f}'.format(_loss, _acc))
    return _loss, _acc


def train(sess,  X_data, y_data, X_valid=None, y_valid=None, epochs=1,
          load=False, shuffle=True, batch_size=128, name='model'):
    if load:
        print('\nLoading saved model')
        return saver.restore(sess, 'model/{}'.format(name))

    print('\nTrain model')
    n_sample = X_data.shape[0]
    n_batch = int((n_sample+batch_size-1) / batch_size)
    for epoch in range(epochs):
        print('\nEpoch {0}/{1}'.format(epoch + 1, epochs))

        if shuffle:
            print('\nShuffling data')
            ind = np.arange(n_sample)
            np.random.shuffle(ind)
            X_data = X_data[ind]
            y_data = y_data[ind]

        for batch in range(n_batch):
            print(' batch {0}/{1}'.format(batch + 1, n_batch), end='\r')
            start = batch * batch_size
            end = min(n_sample, start + batch_size)
            sess.run(train_op, feed_dict={x: X_data[start:end],
                                              y: y_data[start:end],
                                              training: True})
        if X_valid is not None:
            evaluate(sess, X_valid, y_valid)

    print('\n Saving model')
    os.makedirs('model', exist_ok=True)
    saver.save(sess, 'model/{}'.format(name))


def predict(sess, X_data, batch_size=128):
    print('\nPredicting')
    n_classes = ybar.get_shape().as_list()[1]

    n_sample = X_data.shape[0]
    n_batch = int((n_sample+batch_size-1) / batch_size)
    yval = np.empty((n_sample, n_classes))

    for batch in range(n_batch):
        print(' batch {0}/{1}'.format(batch + 1, n_batch), end='\r')
        start = batch * batch_size
        end = min(n_sample, start + batch_size)
        y_batch = sess.run(ybar, feed_dict={x: X_data[start:end]})
        yval[start:end] = y_batch
    print()
    return yval


def make_fgsm(sess, X_data, epochs=1, eps=0.01, batch_size=128):
    print('\nMaking adversarials via FGSM')

    n_sample = X_data.shape[0]
    n_batch = int((n_sample + batch_size - 1) / batch_size)
    X_adv = np.empty_like(X_data)

    for batch in range(n_batch):
        print(' batch {0}/{1}'.format(batch + 1, n_batch), end='\r')
        start = batch * batch_size
        end = min(n_sample, start + batch_size)
        adv = sess.run(x_fgsm, feed_dict={
            x: X_data[start:end],
            fgsm_eps: eps,
            fgsm_epochs: epochs})
        X_adv[start:end] = adv
    print()

    return X_adv


print('\nTraining')
train(sess, X_train, y_train, X_valid, y_valid, load=True, epochs=5, name='fgsm_mnist')

# =============================================================================
# print('\nEvaluating on clean data')
# evaluate(sess, X_test, y_test)
# =============================================================================

print('\nGenerating adversarial data')
X_adv = make_fgsm(sess, X_test, eps=0.02, epochs=6)

# =============================================================================
# print('\nEvaluating on adversarial data')
# evaluate(sess, X_adv, y_test)
# =============================================================================

print('\nRandomly sample adversarial data from each category')
y1 = predict(sess, X_test)
y2 = predict(sess, X_adv)

z0 = np.argmax(y_test, axis=1)
z1 = np.argmax(y1, axis=1)
z2 = np.argmax(y2, axis=1)

X_tmp = np.empty((10, 28, 28))
y_tmp = np.empty((10, 10))
for i in range(10):
    print('Target {0}'.format(i))
    ind, = np.where(np.all([z0 == i, z1 == i, z2 != i], axis=0))
    cur = np.random.choice(ind)
    X_tmp[i] = np.squeeze(X_adv[cur])
    y_tmp[i] = y2[cur]

print('\nPlotting results')

fig = plt.figure(figsize=(10, 1.2))
gs = gridspec.GridSpec(1, 10, wspace=0.05, hspace=0.05)

label = np.argmax(y_tmp, axis=1)
proba = np.max(y_tmp, axis=1)
for i in range(10):
    ax = fig.add_subplot(gs[0, i])
    ax.imshow(X_tmp[i], cmap='gray', interpolation='none')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel('{0} ({1:.2f})'.format(label[i], proba[i]),
                  fontsize=12)

print('\nSaving figure')

gs.tight_layout(fig)
os.makedirs('img', exist_ok=True)
plt.savefig('img/fgsm_mnist.png')
