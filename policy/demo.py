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
from util import *

csv_path = '../../../driving_log.csv'  # my data (fantastic graphic mode)


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
# plt.show()


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

if __name__ == '__main__':

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



    # to avoid " 'NoneType' object has no attribute 'TF_DeleteStatus' " error
    gc.collect()
    K.clear_session()
