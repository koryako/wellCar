import data_prep
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, ELU, Activation, BatchNormalization
from keras.layers.convolutional import Convolution2D, MaxPooling2D


def get_model():
    input_shape = (data_prep.new_height, data_prep.new_width, data_prep.depth)

    model = Sequential()
    model.add(Convolution2D(8, 7, 7, input_shape=input_shape))
    model.add(MaxPooling2D((2, 2)))
    model.add(ELU())
    model.add(BatchNormalization())
    model.add(Convolution2D(8, 5, 5))
    model.add(MaxPooling2D((2, 2)))
    model.add(ELU())
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(512))
    model.add(BatchNormalization())
    model.add(ELU())
    model.add(Dense(256))
    model.add(BatchNormalization())
    model.add(ELU())
    model.add(Dense(128))
    model.add(BatchNormalization())
    model.add(ELU())
    model.add(Dense(1))
    model.add(Activation('tanh'))

    return model
