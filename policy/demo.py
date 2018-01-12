# -*- coding: utf-8 -*-

import numpy as np
#import tensorflow as tf


import os, sys


from keras.models import Sequential, Model
from keras.layers.core import Dense, Dropout, Activation,Lambda
from keras.optimizers import Adam
from keras.utils import np_utils
from keras.layers import Convolution2D, MaxPooling2D, Flatten, Input, ELU
#from keras import initializations
from keras.models import load_model, model_from_json
from keras.layers.normalization import BatchNormalization

from keras import backend as K
import json
import gc
from util import *


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




def main():
    batch_size = 256
    epoch = 10
    csv_path = '../../datasets/run/driving_log.csv'
    center_db,left_db,right_db,steer_db,img_valid,steer_valid=load_csv(csv_path)
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


if __name__ == '__main__':
    load_csv_withPandas( r"/Users/mac/Desktop/ml/datasets/run")

    #Self-Driving-Car-master
    #DDPG-Keras-Torcs