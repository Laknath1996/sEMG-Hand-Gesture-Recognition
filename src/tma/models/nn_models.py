"""
objective : deep learning model architectures for classifying TMA maps
author(s) : Ashwin de Silva, Malsha Perera
date      : 14 Aug 2019
"""

from keras.models import Sequential, load_model
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Dense, Activation, Flatten
from keras.optimizers import sgd, Adam
from keras.engine import InputLayer


def cnn(shape, no_classes):
    """convolutional neural network model"""

    model = Sequential()

    model.add(Conv2D(4, 3, padding='same', input_shape=shape))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(8, 3, padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(100))
    model.add(Activation('relu'))

    model.add(Dense(20))
    model.add(Activation('relu'))

    if no_classes > 2:
        model.add(Dense(no_classes))
        model.add(Activation('softmax'))
        model.compile(loss='categorical_crossentropy', optimizer=sgd(lr=0.01), metrics=['categorical_accuracy'])
    else:
        model.add(Dense(no_classes - 1))
        model.add(Activation('sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer=sgd(lr=0.01), metrics=['accuracy'])

    return model


def nn(shape, no_classes):
    """neural network model"""

    model = Sequential()

    model.add(InputLayer(input_shape=shape))
    model.add(Dense(100))
    model.add(Activation('relu'))

    model.add(Dense(20))
    model.add(Activation('relu'))

    model.add(Dense(no_classes))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy', optimizer=sgd(lr=0.01), metrics=['categorical_accuracy'])

    return model
