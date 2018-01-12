#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 13 15:32:35 2017

@author: priyankadwivedi
"""

## Imports
import glob
import os

import csv
import numpy as np
import cv2
import random


import matplotlib.pyplot as plt
import matplotlib.image as mpimg


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


# compile and train the model using the generator function
train_generator = train_generator(train_samples, batch_size=batch_size)
validation_generator = valid_generator(validation_samples, batch_size=batch_size)

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Lambda, ELU, Activation
from keras.layers.convolutional import Convolution2D, Cropping2D, ZeroPadding2D, MaxPooling2D
from keras.optimizers import SGD, Adam, RMSprop



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



print(model.summary())

