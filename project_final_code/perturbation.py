import tensorflow as tf
import numpy as np
from tensorflow.python.platform import gfile
from sklearn.model_selection import train_test_split
import os.path
import matplotlib.pyplot as plt
from universal_pert import universal_perturbation
tf.reset_default_graph()

classes = [0,1,2,3]
data_complexity = 4
batch_amount = 1
image_batch = 1000
k = 4
delta = 0.3
xi = 1.2
layer = 15
datasets_name = "npy_datasets-%d"%data_complexity
model_name = 'model-c4-l%d-acc1.000000.pb'%layer
image_size = 28
num_train = 100
max_iter_uni = 10

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
x_train, x_test, y_train, y_test = train_test_split(images_sum, labels_sum, test_size=0.9, random_state=50)


num_classes = 2

def jacobian(y_flat, x, inds):
    n = num_classes # Not really necessary, just a quick fix.
    loop_vars = [
         tf.constant(0, tf.int32),
         tf.TensorArray(tf.float32, size=n),
    ]
    _, jacobian = tf.while_loop(
        lambda j,_: j < n,
        lambda j,result: (j+1, result.write(j, tf.gradients(y_flat[inds[j]], x))),
        loop_vars)
    return jacobian.stack()

def create_mnist_npy(dataset, label, len_batch=1000):
    im_array = np.zeros([len_batch, 28, 28, 1], dtype=np.float32)
    counter = int(len_batch/10)
    _class = 0
    s = 0
    for i in range(len(dataset)):
        if label[i] == _class:
            im_array[s] = dataset[i]
            s += 1
            counter -= 1
            if counter == 0:
                counter = int(len_batch/10)
                _class += 1
                if _class == 10:
                    return im_array#np.asarray(im_array, np.float32)

    
    
if __name__ == '__main__':
        print(str(xi))
        persisted_sess = tf.Session()
        model = "model/" + model_name
        # Load the Inception model
        with gfile.FastGFile(model, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            persisted_sess.graph.as_default()
            tf.import_graph_def(graph_def, name='')
        persisted_sess.run(tf.global_variables_initializer())
# =============================================================================
#         persisted_sess.graph.get_operations()
# =============================================================================

        persisted_input = persisted_sess.graph.get_tensor_by_name("input_x:0")
        persisted_output = persisted_sess.graph.get_tensor_by_name("model/logits/BiasAdd:0")

        print('>> Computing feedforward function...')
        def f(image_inp): return persisted_sess.run(persisted_output, feed_dict={persisted_input:image_inp})

        file_perturbation = os.path.join('data', 'universal.npy')

# =============================================================================
#         if os.path.isfile(file_perturbation) == 0:
# =============================================================================
        if True:
            # TODO: Optimize this construction part!
            print('>> Compiling the gradient tensorflow functions. This might take some time...')
            y_flat = tf.reshape(persisted_output, (-1,))
            inds = tf.placeholder(tf.int32, shape=(num_classes,))
            dydx = jacobian(y_flat,persisted_input,inds)

            print('>> Computing gradient function...')
            def grad_fs(image_inp, indices): return persisted_sess.run(dydx, feed_dict={persisted_input: image_inp, inds: indices}).squeeze(axis=1)

            # Running universal perturbation
            v = universal_perturbation(x_train, f, grad_fs, delta=delta, xi=xi, max_iter_uni=max_iter_uni, num_classes=num_classes)
            # Saving the universal perturbation
            np.save(os.path.join(file_perturbation), v)

        else:
            print('>> Found a pre-computed universal perturbation! Retrieving it from ", file_perturbation')
            v = np.load(file_perturbation)

        print('>> Testing the universal perturbation on an image')

        # Test the perturbation on the image
        image_original = x_train[k:k+1]
        label_original = np.argmax(f(image_original))
# =============================================================================
#         # Clip the perturbation to make sure images fit in uint8
#         clipped_v = np.clip(image_original[0,:,:,:]+v[0,:,:,:], 0, 255) - np.clip(image_original[0,:,:,:], 0, 255)
# =============================================================================

        image_perturbed = image_original + v
# =============================================================================
#         image_perturbed = image_original + v[0, :, :, :]
# =============================================================================
        label_perturbed = np.argmax(f(image_perturbed))

        # Show original and perturbed image
        plt.figure(1)
        plt.figure(figsize = (10,3))
        plt.suptitle("data complexity = %d, perturbation = %.2f"%(data_complexity,xi))
# =============================================================================
#         plt.gcf().subplots_adjust(right=0.8)
# =============================================================================
        plt.subplot(131)
        image_original = image_original[0, :, :, 0]
        plt.imshow(image_original)
        plt.colorbar()
        plt.title("original image:"+str(label_original))

        plt.subplot(132)
        image_perturbed = image_perturbed[0, :, :, 0]
        plt.imshow(image_perturbed)
        plt.colorbar()
        plt.title("after perturbed:"+str(label_perturbed))
        
        plt.subplot(133)
        v = v[0, :, :, 0]
        plt.imshow(v)
        plt.colorbar()
        plt.title("universal perturbation")
        
        plt.savefig(datasets_name+"/"+model_name+'-per'+str(xi)+'.png')
        plt.show()
        persisted_sess.close()
        