#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 13 15:32:35 2017

@author: priyankadwivedi
"""

## Imports
import glob
import os
from PIL import Image
import csv
import numpy as np
import cv2
import random
import pandas as pd
from pandas import DataFrame
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


## Link to Udacity's sample data
paths = "/home/animesh/Documents/CarND/CarND-Behavioral-Cloning-P3/data/"
os.chdir(r"/home/animesh/Documents/CarND/CarND-Behavioral-Cloning-P3/data/")

## Link to my collected  data
path_mydata = "/home/animesh/Documents/CarND/CarND-Behavioral-Cloning-P3/my_driving"

#Check size of images
new_path = os.path.join(paths, "IMG/", "*.jpg")
for infile in glob.glob(new_path)[:2]:
    im = Image.open(infile)
    print(im.size, im.mode)

# All images are 320x160

# Import as a dataframe and plot steering
df = pd.read_csv('driving_log.csv', header=0)
df.columns = ["center_image", "left_image", "right_image", "steering", "throttle", "break", "speed"]
df.drop(['throttle', 'break', 'speed'], axis = 1, inplace = True)

import seaborn as sns
sns.set(style="whitegrid", color_codes=True)
sns.distplot(df['steering'], kde = False)

print(len(df))

#8036 samples from Udacity

## Oversample left and right turns. Downsample turns close to zero.
straight =[]
left_turn = []
right_turn = []

for i in range(len(df)):
    keep_prob = random.random()
    # Normal right turns - Double by adding small random fluctuations
    if (df["steering"][i] >0.20 and df["steering"][i] <=0.50):

        for j in range(2):
            new_steering = df["steering"][i]*(1.0 + np.random.uniform(-1,1)/100.0)
            right_turn.append([df["center_image"][i], df["left_image"][i], df["right_image"][i], new_steering])

    # Normal left turns -  Double by adding small random fluctuations

    elif (df["steering"][i] >= -0.50 and df["steering"][i] < -0.15):

        for j in range(2):
            new_steering = df["steering"][i]*(1.0 + np.random.uniform(-1,1)/100.0)
            left_turn.append([df["center_image"][i], df["left_image"][i], df["right_image"][i], new_steering])

    ## Zero angle steering - undersample by 10% worked best
    elif (df["steering"][i] > -0.02 and df["steering"][i] < 0.02):
        if keep_prob <=0.90:
            straight.append([df["center_image"][i], df["left_image"][i], df["right_image"][i], df["steering"][i]])

    else:
        straight.append([df["center_image"][i], df["left_image"][i], df["right_image"][i], df["steering"][i]])

# Create a new list
new_list = []
new_list = right_turn + left_turn + straight
print(len(new_list), len(straight), len(left_turn), len(right_turn))

# Plot new distribution of steering
df_straight = pd.DataFrame(straight, columns=["center_image", "left_image", "right_image", "steering"])
df_left = pd.DataFrame(left_turn, columns=["center_image", "left_image", "right_image", "steering"])
df_right = pd.DataFrame(right_turn, columns=["center_image", "left_image", "right_image", "steering"])

mod_df = pd.concat([df_right, df_left, df_straight], ignore_index=True)
sns.distplot(mod_df['steering'], kde = False)

#Shuffle new_list
random.shuffle(new_list)

# Get my data from training around dirt road
file2 = os.path.join(path_mydata, 'driving_log_mydata.csv')

df_mydata = pd.read_csv(file2, header=0)
df_mydata.columns = ["center_image", "left_image", "right_image", "steering", "throttle", "break", "speed"]
df_mydata.drop(['throttle', 'break', 'speed'], axis = 1, inplace = True)

#Plot my data
sns.distplot(df_mydata['steering'], kde = False)

print(len(df_mydata))

#Oversample my turns - Create 4 turns with small fluctuations and add to existing sample
my_turns = []

for i in range(len(df_mydata)):
    for j in range(4):
        new_steering = df_mydata["steering"][i] * (1.0 + np.random.uniform(-1, 1) / 100.0)
        my_turns.append([df_mydata["center_image"][i], df_mydata["left_image"][i], df_mydata["right_image"][i], new_steering])

print(len(my_turns))
#print(my_turns[0])
final_list = []
final_list =new_list + my_turns
print(len(final_list))

#Shuffle final list
random.shuffle(final_list)

# Break into training and validation samples
from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(final_list, test_size=0.20)

print(len(train_samples), len(validation_samples))

batch_size = 128

# Start with train generator shared in the class and add image augmentations
def train_generator(samples, batch_size=batch_size):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        from sklearn.utils import shuffle
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            # Read center, left and right images from a folder containing Udacity data and my data
            for batch_sample in batch_samples:
                center_name = '/home/animesh/Documents/CarND/CarND-Behavioral-Cloning-P3/data2/IMG/'+batch_sample[0].split('/')[-1]
                center_image = cv2.imread(center_name)
                center_image = cv2.cvtColor(center_image, cv2.COLOR_BGR2RGB)
                left_name = '/home/animesh/Documents/CarND/CarND-Behavioral-Cloning-P3/data2/IMG/'+batch_sample[1].split('/')[-1]
                left_image = cv2.imread(left_name)
                left_image = cv2.cvtColor(left_image, cv2.COLOR_BGR2RGB)
                right_name = '/home/animesh/Documents/CarND/CarND-Behavioral-Cloning-P3/data2/IMG/'+batch_sample[2].split('/')[-1]
                right_image = cv2.imread(right_name)
                right_image = cv2.cvtColor(right_image, cv2.COLOR_BGR2RGB)

                center_angle = float(batch_sample[3])

                # Apply correction for left and right steering
                correction = 0.20
                left_angle = center_angle + correction
                right_angle = center_angle - correction

                # Randomly include either center, left or right image
                num = random.random()
                if num <= 0.33:
                    select_image = center_image
                    select_angle = center_angle
                    images.append(select_image)
                    angles.append(select_angle)
                elif num>0.33 and num<=0.66:
                    select_image = left_image
                    select_angle = left_angle
                    images.append(select_image)
                    angles.append(select_angle)
                else:
                    select_image = right_image
                    select_angle = right_angle
                    images.append(select_image)
                    angles.append(select_angle)

                # Randomly horizontally flip selected images with 80% probability
                keep_prob = random.random()
                if keep_prob >0.20:
                    flip_image = np.fliplr(select_image)
                    flip_angle = -1*select_angle
                    images.append(flip_image)
                    angles.append(flip_angle)

                # Augment with images of different brightness
                # Randomly select a percent change
                change_pct = random.uniform(0.4, 1.2)

                # Change to HSV to change the brightness V
                hsv = cv2.cvtColor(select_image, cv2.COLOR_RGB2HSV)

                hsv[:, :, 2] = hsv[:, :, 2] * change_pct
                # Convert back to RGB and append

                bright_img = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
                images.append(bright_img)
                angles.append(select_angle)

                ## Randomly shear image with 80% probability
                shear_prob = random.random()
                if shear_prob >=0.20:
                    shear_range = 40
                    rows, cols, ch = select_image.shape
                    dx = np.random.randint(-shear_range, shear_range + 1)
                    #    print('dx',dx)
                    random_point = [cols / 2 + dx, rows / 2]
                    pts1 = np.float32([[0, rows], [cols, rows], [cols / 2, rows / 2]])
                    pts2 = np.float32([[0, rows], [cols, rows], random_point])
                    dsteering = dx / (rows / 2) * 360 / (2 * np.pi * 25.0) / 10.0
                    M = cv2.getAffineTransform(pts1, pts2)
                    shear_image = cv2.warpAffine(center_image, M, (cols, rows), borderMode=1)
                    shear_angle = select_angle + dsteering
                    images.append(shear_image)
                    angles.append(shear_angle)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)

            yield shuffle(X_train, y_train)

def valid_generator(samples, batch_size=batch_size):
        num_samples = len(samples)
        while 1:  # Loop forever so the generator never terminates
            from sklearn.utils import shuffle
            shuffle(samples)
            for offset in range(0, num_samples, batch_size):
                batch_samples = samples[offset:offset + batch_size]

                images = []
                angles = []

                #Validation generator only has center images and no augmentations
                for batch_sample in batch_samples:
                    center_name = '/home/animesh/Documents/CarND/CarND-Behavioral-Cloning-P3/data2/IMG/' + batch_sample[0].split('/')[-1]
                    center_image = cv2.imread(center_name)
                    center_image = cv2.cvtColor(center_image, cv2.COLOR_BGR2RGB)

                    center_angle = float(batch_sample[3])

                    images.append(center_image)
                    angles.append(center_angle)

                X_train = np.array(images)
                y_train = np.array(angles)

                yield shuffle(X_train, y_train)


# compile and train the model using the generator function
train_generator = train_generator(train_samples, batch_size=batch_size)
validation_generator = valid_generator(validation_samples, batch_size=batch_size)

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Lambda, ELU, Activation
from keras.layers.convolutional import Convolution2D, Cropping2D, ZeroPadding2D, MaxPooling2D
from keras.optimizers import SGD, Adam, RMSprop

# Function to resize image to 64x64
def resize_image(image):
    import tensorflow as tf
    return tf.image.resize_images(image,[64,64])


#Params
row, col, ch = 160, 320, 3
nb_classes = 1


model = Sequential()
model.add(ZeroPadding2D((1, 1), input_shape=(row, col, ch)))
# Crop pixels from top and bottom of image
model.add(Cropping2D(cropping=((60, 20), (0, 0))))

# Resise data within the neural network
model.add(Lambda(resize_image))
# Normalize data
model.add(Lambda(lambda x: (x / 127.5 - 1.)))

# First convolution layer so the model can automatically figure out the best color space for the hypothesis
model.add(Convolution2D(3, 1, 1, border_mode='same', name='color_conv'))

# CNN model

model.add(Convolution2D(32, 3,3 ,border_mode='same', subsample=(2,2), name='conv1'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2),strides=(1,1), name='pool1'))

model.add(Convolution2D(64, 3,3 ,border_mode='same',subsample=(2,2), name='conv2'))
model.add(Activation('relu',name='relu2'))
model.add(MaxPooling2D(pool_size=(2,2), name='pool2'))

model.add(Convolution2D(128, 3,3,border_mode='same',subsample=(1,1), name='conv3'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size= (2,2), name='pool3'))

model.add(Flatten())
model.add(Dropout(0.5))

model.add(Dense(128, name='dense1'))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(128, name='dense2'))

model.add(Dense(1,name='output'))

model.compile(optimizer=Adam(lr= 0.0001), loss="mse")

# weights_path = '/home/animesh/Documents/CarND/CarND-Behavioral-Cloning-P3/model3.h5'
# model.load_weights(weights_path)
#
# # Make all conv layers non trainable
# for layer in model.layers[:16]:
#     layer.trainable = False
#
# model.compile(optimizer=Adam(lr= 1e-5), loss="mse")

nb_epoch = 8
samples_per_epoch = 20000
nb_val_samples = 2000


#save every model using Keras checkpoint
from keras.callbacks import ModelCheckpoint
filepath="/home/animesh/Documents/CarND/CarND-Behavioral-Cloning-P3/checkpoint2/check-{epoch:02d}-{val_loss:.4f}.hdf5"
checkpoint = ModelCheckpoint(filepath= filepath, verbose=1, save_best_only=False)
callbacks_list = [checkpoint]

#Model fit generator
history_object = model.fit_generator(train_generator, samples_per_epoch= samples_per_epoch,
                                     validation_data=validation_generator,
                                     nb_val_samples=nb_val_samples, nb_epoch=nb_epoch, verbose=1, callbacks=callbacks_list)

#Plot losses
import matplotlib.pyplot as plt

print(history_object.history.keys())
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])

plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

#Save model
from keras.models import model_from_json

model_json = model.to_json()
with open("/home/animesh/Documents/CarND/CarND-Behavioral-Cloning-P3/model_final.json", "w") as json_file:
    json_file.write(model_json)

model.save("/home/animesh/Documents/CarND/CarND-Behavioral-Cloning-P3/model_final.h5")
print("Saved model to disk")

print(model.summary())



import numpy as np
import tensorflow as tf
import csv
import matplotlib.pyplot as plt
import os, sys
import cv2
from sklearn.model_selection import train_test_split
from keras.models import Sequential, Model
from keras.layers.core import Dense, Dropout, Activation,Lambda
from keras.optimizers import Adam
from keras.utils import np_utils
from keras.layers import Convolution2D, MaxPooling2D, Flatten, Input, ELU
from keras import initializations
from keras.models import load_model, model_from_json
from keras.layers.normalization import BatchNormalization
from sklearn.utils import shuffle
from keras import backend as K
import json
import gc

csv_path = '../../../run1/driving_log.csv'  # my data (fantastic graphic mode)
csv_path1 = 'data/driving_log.csv'  # udacity data (fastest graphic mode)

center_db, left_db, right_db, steer_db = [], [], [], []
Rows, Cols = 64, 64
offset = 0.22

# read csv file
with open(csv_path) as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        if float(row['steering']) != 0.0:
            center_db.append(row['center'])
            left_db.append(row['left'].strip())
            right_db.append(row['right'].strip())
            steer_db.append(float(row['steering']))
        else:
            prob = np.random.uniform()
            if prob <= 0.15:
                center_db.append(row['center'])
                left_db.append(row['left'].strip())
                right_db.append(row['right'].strip())
                steer_db.append(float(row['steering']))

# shuffle a dataset
center_db, left_db, right_db, steer_db = shuffle(center_db, left_db, right_db, steer_db)

# split train & valid data
img_train, img_valid, steer_train, steer_valid = train_test_split(center_db, steer_db, test_size=0.1, random_state=42)

plt.hist(steer_db, bins= 50, color= 'orange')
plt.xlabel('steering value')
plt.ylabel('counts')
plt.show()

def select_img(center, left, right, steer, num, offsets=0.22):
    """
    randomly select among center, left, right images

    add ±0.22 to left, right steering angle.
    couldn't find exact left, right steering angle by using geometric method because we didn't have enough information.
    """
    rand = np.random.randint(3)

    if rand == 0:
        image, steering = cv2.imread(center[num]), steer[num]
    elif rand == 1:
        image, steering = cv2.imread(left[num]), steer[num] + offsets
    elif rand == 2:
        image, steering = cv2.imread(right[num]), steer[num] - offsets
    if abs(steering) > 1:
        steering = -1 if (steering < 0) else 1

    return image, steering

def valid_img(valid_image, valid_steer, num):
    """ using only center image for validation """
    steering = valid_steer[num]
    image = cv2.imread(valid_image[num])
    return image, steering

def crop_img(image):
    """ crop unnecessary parts """
    cropped_img = image[63:136, 0:319]
    resized_img = cv2.resize(cropped_img, (Cols, Rows), cv2.INTER_AREA)
    img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)
    return resized_img

def shift_img(image, steer):
    """
    randomly shift image horizontally
    add proper steering angle to each image
    """
    max_shift = 55
    max_ang = 0.14  # ang_per_pixel = 0.0025

    rows, cols, _ = image.shape

    random_x = np.random.randint(-max_shift, max_shift + 1)
    dst_steer = steer + (random_x / max_shift) * max_ang
    if abs(dst_steer) > 1:
        dst_steer = -1 if (dst_steer < 0) else 1

    mat = np.float32([[1, 0, random_x], [0, 1, 0]])
    dst_img = cv2.warpAffine(image, mat, (cols, rows))
    return dst_img, dst_steer

def brightness_img(image):
    """
    randomly change brightness by converting Y value
    """
    br_img = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    coin = np.random.randint(2)
    if coin == 0:
        random_bright = 0.2 + np.random.uniform(0.2, 0.6)
        br_img[:, :, 2] = br_img[:, :, 2] * random_bright
    br_img = cv2.cvtColor(br_img, cv2.COLOR_HSV2RGB)
    return br_img

def generate_shadow(image, min_alpha=0.5, max_alpha = 0.75):
    """generate random shadow in random region"""

    top_x, bottom_x = np.random.randint(0, Cols, 2)
    coin = np.random.randint(2)
    rows, cols, _ = image.shape
    shadow_img = image.copy()
    if coin == 0:
        rand = np.random.randint(2)
        vertices = np.array([[(50, 65), (45, 0), (145, 0), (150, 65)]], dtype=np.int32)
        if rand == 0:
            vertices = np.array([[top_x, 0], [0, 0], [0, rows], [bottom_x, rows]], dtype=np.int32)
        elif rand == 1:
            vertices = np.array([[top_x, 0], [cols, 0], [cols, rows], [bottom_x, rows]], dtype=np.int32)
        mask = image.copy()
        channel_count = image.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (0,) * channel_count
        cv2.fillPoly(mask, [vertices], ignore_mask_color)
        rand_alpha = np.random.uniform(min_alpha, max_alpha)
        cv2.addWeighted(mask, rand_alpha, image, 1 - rand_alpha, 0., shadow_img)

    return shadow_img

def flip_img(image, steering):
    """ randomly flip image to gain right turn data (track1 is biaed in left turn) """
    flip_image = image.copy()
    flip_steering = steering
    num = np.random.randint(2)
    if num == 0:
        flip_image, flip_steering = cv2.flip(image, 1), -steering
    return flip_image, flip_steering

def network_model():
    """
    designed with 4 convolutional layer & 3 fully connected layer
    weight init : glorot_uniform
    activation func : relu
    pooling : maxpooling
    used dropout
    """

    model = Sequential()
    model.add(Lambda(lambda x: x / 127.5 - 1.0, input_shape=(Rows, Cols, 3)))
    model.add(Convolution2D(32, 3, 3, border_mode='same', subsample=(2, 2), activation='relu', name='Conv1'))
    #model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), strides=None, border_mode='same'))
    model.add(Convolution2D(64, 3, 3, border_mode='same', subsample=(2, 2), activation='relu', name='Conv2'))
    #model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), strides=None, border_mode='same'))
    model.add(Convolution2D(128, 3, 3, border_mode='same', subsample=(1, 1), activation='relu', name='Conv3'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=None, border_mode='same'))
    #model.add(BatchNormalization())
    model.add(Convolution2D(128, 2, 2, border_mode='same', subsample=(1, 1), activation='relu', name='Conv4'))
    #model.add(BatchNormalization())
    model.add(Flatten())
    model.add(Dropout(0.2))
    model.add(Dense(128, activation='relu', name='FC1'))
    model.add(Dropout(0.5))
    model.add(Dense(128, activation='relu', name='FC2'))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu', name='FC3'))
    model.add(Dense(1))
    model.summary()
    return model

def generate_train(center, left, right, steer):
    """
    data augmentation
    transformed image & crop
    """

    num = np.random.randint(0, len(steer))
    # to avoid bias in straight angle
    #bal = True
    #while bal:
    #    num = np.random.randint(0, len(steer))
    #    check_steer = steer[num]
    #    if check_steer == 0:
    #        rand = np.random.uniform()
    #        if rand <= 0.25:
    #            bal = False
    #    else:
    #        bal = False

    image, steering = select_img(center, left, right, steer, num, offset)

    image, steering = shift_img(image, steering)
    image, steering = flip_img(image, steering)
    image = brightness_img(image)
    # image = generate_shadow(image)
    image = crop_img(image)
    return image, steering

def generate_valid(img_valid, steer_valid):
    """ generate validation set """
    img_set = np.zeros((len(img_valid), Rows, Cols, 3))
    steer_set = np.zeros(len(steer_valid))

    for i in range(len(img_valid)):
        img, steer = valid_img(img_valid, steer_valid, i)
        img_set[i] = crop_img(img)

        steer_set[i] = steer
    return img_set, steer_set

def generate_train_batch(center, left, right, steering, batch_size):
    """ compose training batch set """
    image_set = np.zeros((batch_size, Rows, Cols, 3))
    steering_set = np.zeros(batch_size)

    while 1:
        for i in range(batch_size):
            img, steer = generate_train(center, left, right, steering)
            image_set[i] = img
            steering_set[i] = steer
        yield image_set, steering_set

batch_size = 256
epoch = 10

train_generator = generate_train_batch(center_db, left_db, right_db, steer_db, batch_size)
image_val, steer_val = generate_valid(img_valid, steer_valid)

model = network_model()

adam = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
model.compile(optimizer=adam, loss='mse')

model_json = 'model.json'
model_weights = 'model.h5'

history = model.fit_generator(train_generator, samples_per_epoch=20480, nb_epoch=epoch,
                              validation_data=(image_val, steer_val), verbose=1)

json_string = model.to_json()

try:
    os.remove(model_json)
    os.remove(model_weights)
except OSError:
    pass

with open(model_json, 'w') as jfile:
    json.dump(json_string, jfile)
model.save_weights(model_weights)

# to avoid " 'NoneType' object has no attribute 'TF_DeleteStatus' " error
gc.collect()
K.clear_session()

