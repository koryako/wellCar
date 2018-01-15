# -*- coding: utf-8 -*-
from keras.layers import Dense, Activation, Reshape, Merge
from keras.layers.core import Flatten, Reshape, Dropout, Lambda
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D, AveragePooling1D
from keras.optimizers import Adam, SGD
from keras.utils.np_utils import to_categorical
from keras.callbacks import Callback as KerasCallback
import cv2
import random as rand
from numpy import genfromtxt
import ntpath
import numpy as np
#import tensorflow as tf
#tf.python.control_flow_ops = tf
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
import gc
from util import *
from keras.preprocessing.image import ImageDataGenerator# keras 数据批量生成
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
    batch_size = 250
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
    #x_train, x_validation, y_train, y_validation = data_prep.prep_data()

    #datagen = ImageDataGenerator()
    #model = model_creation.get_model()
    #model.compile('adam', 'mean_squared_error', ['accuracy'])
    #history = model.fit_generator(datagen.flow(x_train, y_train, batch_size=32), samples_per_epoch=len(x_train),
                              #nb_epoch=5, validation_data=datagen.flow(x_validation, y_validation, batch_size=32),
                              #nb_val_samples=len(x_validation))
    #print(history)
    #history = model.fit_generator(train_generator, steps_per_epoch=100, nb_epoch=epoch,
                              #validation_data=(image_val, steer_val), verbose=1)
    history = model.fit_generator(train_generator, steps_per_epoch=200, nb_epoch=epoch,
                              validation_data=(image_val, steer_val), verbose=1)
    
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

def main_more():
    #loading needed datasets

    #datasets with straight steering
    datasets = ['record-backward', 'record-forward', 'record-backward-2', 'record-forward-2']
    #datasets=['record-backward']
    steering, center_camera = load_datasets (datasets)

    #datasets with recovering from left divergence
    steering_lr, center_camera_lr = load_datasets ([
        #strong divergence, using only 1/4 of the dataset
        ('record-left-recover', 0.5),
        #medium divergence
        ('record-left-recover-2', 1),
        #light divergence
        ('record-left-recover-3', 1)
    ])

    #datasets with recovering from right divergence
    steering_rr, center_camera_rr = load_datasets ([
        ('record-right-recover', 0.5),
        ('record-right-recover-2', 1),
        ('record-right-recover-3', 1),
    ])


    #remove steering that less than 0 for recover from left dataset
    #so removing divergence steering and staying only recovering steering
    arrs = sparse_arrays_by_first_array_value (0, lambda v: v < 0, 0, [steering_lr, center_camera_lr])
    steering_lr, center_camera_lr = arrs

    #remove steering that greater than 0 for recover dataset
    arrs = sparse_arrays_by_first_array_value (0, lambda v: v > 0, 0, [steering_rr, center_camera_rr])
    steering_rr, center_camera_rr = arrs

    #joining datasets
    steering = np.concatenate ([steering, steering_lr, steering_rr])
    center_camera = np.concatenate ([center_camera, center_camera_lr, center_camera_rr])

    #We discussed proj 3 in our study group and I decided to filter steering wheel values around zero,
    #as it dominates in the dataset
    arrs = sparse_arrays_by_first_array_value (0.1, lambda v: abs(v) < 1e-5, 0, [steering, center_camera])
    steering, center_camera = arrs

    dataset_images = load_images (center_camera)


    model = Sequential ([
        Reshape ((160, 320, 1), input_shape=(160, 320)),
        
        Convolution2D (24, 8, 8, border_mode='valid'),
        MaxPooling2D (pool_size=(2, 2)),
        Dropout (0.5),
        Activation ('relu'),

        #77x157
        Convolution2D (36, 5, 5, border_mode='valid'),
        MaxPooling2D (pool_size=(2, 2)),
        Dropout (0.5),
        Activation ('relu'),

        #37x77
        Convolution2D (48, 5, 5, border_mode='valid'),
        MaxPooling2D (pool_size=(2, 2)),
        Dropout (0.5),
        Activation ('relu'),

        #17x37
        Convolution2D (64, 3, 3, border_mode='valid'),
        MaxPooling2D (pool_size=(2, 2)),
        Dropout (0.5),
        Activation ('relu'),

        #8x18
        Convolution2D (64, 2, 2, border_mode='valid'),
        MaxPooling2D (pool_size=(2, 2)),
        Dropout (0.5),
        Activation ('relu'),

        #4x9
        Flatten (),
        
        Dense (1024),
        Dropout (0.5),
        Activation ('relu'),

        Dense (512),
        Dropout (0.5),
        Activation ('relu'),

        Dense (256),
        Activation ('relu'),
        
        Dense (128),
        Activation ('relu'),
        
        Dense (32),
        Activation ('tanh'),
        
        Dense (1)
    ])

    optimizer = Adam (lr=1e-4)

    model.compile (
         optimizer=optimizer,
         loss='mse',
         metrics=[]
    )


    #training

    #splitting into train and validation dataset
    train, valid = train_test_split ([dataset_images, steering], test_size=0.33)
    train_dataset_images, train_steering = train
    valid_dataset_images, valid_steering = valid

    print (len(train_dataset_images), len (valid_dataset_images))

    batch_size = 112
    epochs = 30

    class SaveModel(KerasCallback):
        def on_epoch_end(self, epoch, logs={}):
            epoch += 1
            if (epoch>9):
                with open ('model-' + str(epoch) + '.json', 'w') as file:
                    file.write (model.to_json ())
                    file.close ()

                model.save_weights ('model-' + str(epoch) + '.h5')

    model.fit_generator (
        augment_generator(train_dataset_images, train_steering, batch_size),
        samples_per_epoch=10*112,
        nb_epoch=epochs,
        validation_data=(valid_dataset_images, valid_steering),
        callbacks = [SaveModel ()]
    )

    steering_test, center_camera_test = load_datasets (['record-test'])
    images_test = load_images (center_camera_test)
    loss = model.evaluate (images_test, steering_test)
    print ('test loss: ', loss)
    
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

"""