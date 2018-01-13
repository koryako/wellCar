# -*- coding: utf-8 -*-

import numpy as np
#import tensorflow as tf


import os, sys

from vis import show_loss
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
    batch_size = 25
    epoch = 10
    #csv_path = '../../datasets/run/driving_log.csv'
    csv_path = '../../../run1/driving_log.csv'
    center_db,left_db,right_db,steer_db,img_valid,steer_valid=load_csv(csv_path)
    train_generator = generate_train_batch(center_db, left_db, right_db, steer_db, batch_size)
    image_val, steer_val = generate_valid(img_valid, steer_valid)
    
    model = network_model()
    
    adam = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(optimizer=adam, loss='mse')
    
    model_json = 'model.json'
    model_weights = 'model.h5'
    
    history = model.fit_generator(train_generator, steps_per_epoch=100, nb_epoch=epoch,
                              validation_data=(image_val, steer_val), verbose=1)
    #history = model.fit_generator(train_generator, samples_per_epoch=20480, nb_epoch=epoch,
                              #validation_data=(image_val, steer_val), verbose=1)
    
    json_string = model.to_json()

    try:
        os.remove(model_json)
        os.remove(model_weights)
    except OSError:
        pass

    saveModel(model,model_json,model_weights)
    show_loss(history)
    # to avoid " 'NoneType' object has no attribute 'TF_DeleteStatus' " error
    gc.collect()
    K.clear_session()


if __name__ == '__main__':
    #load_csv_withPandas( r"/Users/mac/Desktop/ml/datasets/run")
    main()
    #Self-Driving-Car-master
    #DDPG-Keras-Torcs
    #steering-a-car-behavioral-cloning
"""
    batch_size = 128

# compile and train the model using the generator function
train_generator = train_generator(train_samples, batch_size=batch_size)
validation_generator = valid_generator(validation_samples, batch_size=batch_size)

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Lambda, ELU, Activation
from keras.layers.convolutional import Convolution2D, Cropping2D, ZeroPadding2D, MaxPooling2D
from keras.optimizers import SGD, Adam, RMSprop





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
"""