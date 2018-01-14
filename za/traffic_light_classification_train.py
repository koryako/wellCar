#!/usr/bin/python2
import sys, os, time
import itertools
import math, random
import glob
import tensorflow as tf
import numpy as np
import cv2
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from IPython.display import Image, display
import warnings
warnings.filterwarnings('ignore')

max_epochs = 25
base_image_path = "dataset/traffic_light_images/"
image_types = ["red", "green", "yellow"]
input_img_x = 32
input_img_y = 32
train_test_split_ratio = 0.9
batch_size = 32
checkpoint_name = "model.ckpt"

# Helper layer functions
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W, stride):
    return tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')
"""
x = tf.placeholder(tf.float32, shape=[None, input_img_x, input_img_y, 3])
y_ = tf.placeholder(tf.float32, shape=[None, len(image_types)])
x_image = x

# Our first three convolutional layers, of 16 3x3 filters
W_conv1 = weight_variable([3, 3, 3, 16])
b_conv1 = bias_variable([16])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1, 1) + b_conv1)

W_conv2 = weight_variable([3, 3, 16, 16])
b_conv2 = bias_variable([16])
h_conv2 = tf.nn.relu(conv2d(h_conv1, W_conv2, 1) + b_conv2)

W_conv3 = weight_variable([3, 3, 16, 16])
b_conv3 = bias_variable([16])
h_conv3 = tf.nn.relu(conv2d(h_conv2, W_conv3, 1) + b_conv3)

# Our pooling layer

h_pool4 = max_pool_2x2(h_conv3)

n1, n2, n3, n4 = h_pool4.get_shape().as_list()

W_fc1 = weight_variable([n2*n3*n4, 3])
b_fc1 = bias_variable([3])

# We flatten our pool layer into a fully connected layer

h_pool4_flat = tf.reshape(h_pool4, [-1, n2*n3*n4])
print h_pool4_flat.shape
y = tf.matmul(h_pool4_flat, W_fc1) + b_fc1

sess = tf.InteractiveSession()
# Our loss function and optimizer

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, y_))
train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)


sess.run(tf.initialize_all_variables())

saver = tf.train.Saver()
time_start = time.time()

v_loss = least_loss = 99999999

# Load data
"""
def load_img_to_data(base_image_path,image_types):
    full_set = []
    for im_type in image_types:
        for ex in glob.glob(os.path.join(base_image_path, im_type, "*")):
            im = cv2.imread(ex)
            if not im is None:
                im = cv2.resize(im, (32, 32))

                # Create an array representing our classes and set it
                one_hot_array = [0] * len(image_types)
                one_hot_array[image_types.index(im_type)] = 1
                assert(im.shape == (32, 32, 3))

                full_set.append((im, one_hot_array, ex))

    random.shuffle(full_set)

    # We split our data into a training and test set here

    split_index = int(math.floor(len(full_set) * train_test_split_ratio))
    train_set = full_set[:split_index]
    test_set = full_set[split_index:]

    # We ensure that our training and test sets are a multiple of batch size
    train_set_offset = len(train_set) % batch_size
    test_set_offset = len(test_set) % batch_size
    train_set = train_set[: len(train_set) - train_set_offset]
    test_set = test_set[: len(test_set) - test_set_offset]

    train_x, train_y, train_z = zip(*train_set)
    test_x, test_y, test_z = zip(*test_set)
    
    return (train_x, train_y, train_z,test_x, test_y, test_z)

train_x, train_y, train_z,test_x, test_y, test_z=load_img_to_data(base_image_path,image_types)

print("Starting training... [{} training examples]".format(len(train_x)))

"""

v_loss = 9999999
train_loss = []
val_loss = []

for i in range(0, max_epochs):

    # Iterate over our training set
    for tt in range(0, (len(train_x) / batch_size)):
        start_batch = batch_size * tt
        end_batch = batch_size * (tt + 1)
        train_step.run(feed_dict={x: train_x[start_batch:end_batch], y_: train_y[start_batch:end_batch]})
        ex_seen = "Current epoch, examples seen: {:20} / {} \r".format(tt * batch_size, len(train_x))
        sys.stdout.write(ex_seen.format(tt * batch_size))
        sys.stdout.flush()

    ex_seen = "Current epoch, examples seen: {:20} / {} \r".format((tt + 1) * batch_size, len(train_x))
    sys.stdout.write(ex_seen.format(tt * batch_size))
    sys.stdout.flush()

    t_loss = loss.eval(feed_dict={x: train_x, y_: train_y})
    v_loss = loss.eval(feed_dict={x: test_x, y_: test_y})
    
    train_loss.append(t_loss)
    val_loss.append(v_loss)

    sys.stdout.write("Epoch {:5}: loss: {:15.10f}, val. loss: {:15.10f}".format(i + 1, t_loss, v_loss))

    if v_loss < least_loss:
        sys.stdout.write(", saving new best model to {}".format(checkpoint_name))
        least_loss = v_loss
        filename = saver.save(sess, checkpoint_name)

    sys.stdout.write("\n")
    """