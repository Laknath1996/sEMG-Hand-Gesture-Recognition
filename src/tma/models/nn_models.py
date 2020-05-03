"""
The MIT License (MIT)
Copyright (c) 2020, Ashwin De Silva and Malsha Perera

Other Contributors : Asma Naim, Kithmin Wickramasinghe, Thilina
Lalitharatne, Simon Kappel

Permission is hereby granted, free of charge, to any person obtaining a
copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation
the rights to use, copy, modify, merge, publish, distribute, sublicense,
and/or sell copies of the Software, and to permit persons to whom the
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included
in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

objective : Contains the deep learning architectures (nn, cnn)

The code is based on the following paper :
[1] A. D. Silva, M. V. Perera, K. Wickramasinghe, A. M. Naim,
    T. Dulantha Lalitharatne and S. L. Kappel, "Real-Time Hand Gesture
    Recognition Using Temporal Muscle Activation Maps of Multi-Channel
    Semg Signals," ICASSP 2020 - 2020 IEEE International Conference
    on Acoustics, Speech and Signal Processing (ICASSP), Barcelona,
    Spain, 2020, pp. 1299-1303.
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
