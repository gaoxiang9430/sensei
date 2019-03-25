"""
This program is designed to train models for gstrb dataset
Author: Xiang Gao (xiang.gao@us.fujitsu.com)
Time: Sep, 21, 2018
"""

from __future__ import print_function

import os
import glob
import numpy as np
import keras
from keras import applications
from keras.models import Model, Sequential
from keras.layers import Dense, Dropout, Flatten, Input
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as k
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
from keras.layers import MaxPooling2D, Convolution2D, Activation, Dropout, Flatten, Dense, InputLayer
import tensorflow as tf
import cv2
from deepaugment.data_generator import DataGenerator
from deepaugment.util import logger
from deepaugment.config import ExperimentalConfig
from keras.datasets import mnist, fashion_mnist
from keras.constraints import maxnorm
from keras.layers.normalization import BatchNormalization


class FashionMnist:

    """ source dir is the relative path of gtsrb data set' """
    def __init__(self, start_point=0, epoch=50):

        self.config = ExperimentalConfig.gen_config()
        # Config related to images in the gtsrb dataset
        self.start_point = start_point
        (self.x_train, self.y_train), (self.x_test, self.y_test) = fashion_mnist.load_data()

        self.num_classes = len(np.unique(self.y_train))
        self.epoch = epoch
        self.name = "fashionMnist"
        self.batch_size = 512
        self.input_shape = (28, 28, 1)

        self.script_path = os.path.dirname(os.path.abspath(__file__))

        # define default data generator
        self.datagen_rotation = ImageDataGenerator(
            featurewise_center=False,             # set input mean to 0 over the dataset
            samplewise_center=False,              # set each sample mean to 0
            featurewise_std_normalization=False,  # divide inputs by std of dataset
            samplewise_std_normalization=False,   # divide each input by its std
            zca_whitening=False,                  # apply ZCA whitening
            zca_epsilon=1e-06,                    # epsilon for ZCA whitening
            rotation_range=30,                    # randomly rotate images in the range (deg 0 to 180)
            width_shift_range=0.1,                # randomly shift images horizontally
            height_shift_range=0.1,               # randomly shift images vertically
            shear_range=0.2,                      # set range for random shear
            zoom_range=0.,                        # set range for random zoom
            channel_shift_range=0.,               # set range for random channel shifts
            fill_mode='nearest',                  # set mode for filling points outside the input boundaries
            cval=0.,                              # value used for fill_mode = "constant"
            horizontal_flip=True,                 # randomly flip images
            vertical_flip=False,                  # randomly flip images
            rescale=None,                         # set rescaling factor (applied before any other transformation)
            preprocessing_function=None,          # set function that will be applied on each input
            data_format=None,                     # image data format, either "channels_first" or
                                                  # "channels_last"
            validation_split=0.0                  # fraction of images reserved for validation
                                                  # (strictly between 0 and 1)
        )

    def preprocess_original_imgs(self, x):
        for i in range(len(x)):
            x[i] = cv2.resize(x[i], (28, 28))
        x = np.array(x).reshape(-1, 28, 28, 1).astype('float32') / 255.
        return x

    def load_original_test_data(self):
        y_test = keras.utils.to_categorical(self.y_test, self.num_classes)
        return list(self.x_test), y_test

    def load_original_data(self, data_type='train'):
        if data_type == 'train':
            y_train = keras.utils.to_categorical(self.y_train, self.num_classes)
            return list(self.x_train), y_train
        else:  # without validation set
            y_test = keras.utils.to_categorical(self.y_test, self.num_classes)
            return list(self.x_test), y_test

    def def_model(self, model_id=0):
        input_shape = self.input_shape
        if model_id == 0:  # vgg 16
            input_tensor = Input(shape=input_shape)
            x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(input_tensor)
            x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
            x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

            # Block 2
            x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
            x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
            x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

            # Block 3
            x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
            x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
            x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
            x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

            # Block 4
            x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
            x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
            x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
            x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

            # Classification block
            x = Flatten(name='flatten')(x)
            x = Dense(4096, activation='relu', name='fc1')(x)
            x = Dense(4096, activation='relu', name='fc2')(x)
            x = Dropout(0.5)(x)
            x = Dense(10, activation='softmax', name='predictions')(x)
            return_model = Model(inputs=[input_tensor], outputs=[x])
            return_model.compile(loss=keras.losses.categorical_crossentropy,
                                 optimizer=keras.optimizers.Adam(),
                                 metrics=['accuracy'])

        elif model_id == 1:
            """
            Convolutional Neural Network: https://github.com/umbertogriffo/Fashion-mnist-cnn-keras/blob/
            master/src/convolutional/fashion_mnist_cnn.py
            """
            return_model = Sequential()
            return_model.add(Conv2D(32, (5, 5), input_shape=self.input_shape, padding='same', activation='relu'))
            return_model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))

            return_model.add(Conv2D(64, (5, 5), padding='same', activation='relu'))
            return_model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))

            return_model.add(Conv2D(128, (1, 1), padding='same', activation='relu'))
            return_model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))

            return_model.add(Flatten())

            return_model.add(Dense(1024, activation='relu', kernel_constraint=maxnorm(3)))
            return_model.add(Dropout(0.5))
            return_model.add(Dense(512, activation='relu', kernel_constraint=maxnorm(3)))
            return_model.add(Dropout(0.5))

            return_model.add(Dense(self.num_classes, activation='softmax'))
            # Compile model
            lrate = 0.1
            decay = lrate / self.epoch
            sgd = keras.optimizers.SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=True)
            return_model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
        elif model_id == 2:
            # https://github.com/cmasch/zalando-fashion-mnist/blob/master/Simple_Convolutional_Neural_Network_Fashion-MNIST.ipynb
            cnn = Sequential()

            cnn.add(InputLayer(input_shape=self.input_shape))

            cnn.add(BatchNormalization())
            cnn.add(Convolution2D(64, (4, 4), padding='same', activation='relu'))
            cnn.add(MaxPooling2D(pool_size=(2, 2)))
            cnn.add(Dropout(0.1))

            cnn.add(Convolution2D(64, (4, 4), activation='relu'))
            cnn.add(MaxPooling2D(pool_size=(2, 2)))
            cnn.add(Dropout(0.3))

            cnn.add(Flatten())

            cnn.add(Dense(256, activation='relu'))
            cnn.add(Dropout(0.5))

            cnn.add(Dense(64, activation='relu'))
            cnn.add(BatchNormalization())

            cnn.add(Dense(self.num_classes, activation='softmax'))
            cnn.compile(loss='categorical_crossentropy',
                        optimizer=keras.optimizers.Adam(),
                        metrics=['accuracy'])

            return cnn
        elif model_id == 3:
            pass  # https://github.com/markjay4k/Fashion-MNIST-with-Keras/blob/master/pt%204%20-%20Deeper%20CNNs.ipynb
        else:
            raise Exception("unsupported model")
        return return_model

    def train_dnn_model(self, _model=[0, "models/gtsrb"], x_train=None, y_train=None,
                        x_val=None, y_val=None, train_strategy=None):
        """
        train a dnn model on gstrb dataset based on train strategy
        """
        # training config

        tf.keras.backend.clear_session()

        epochs = self.epoch
        batch_size = self.batch_size

        k.set_image_data_format('channels_last')

        model_id = _model[0]
        weights_file = _model[1]
        weights_file = os.path.join(self.script_path, weights_file)

        model = self.def_model(model_id)
        if self.start_point > 0:
            model.load_weights(weights_file)

        checkpoint = ModelCheckpoint(weights_file, monitor='acc', verbose=1,
                                     save_best_only=False, mode='max')
        callbacks_list = [checkpoint]
        x_val = self.preprocess_original_imgs(x_val)
        '''
        if train_strategy is None:
            data = self.datagen_rotation.flow(x_train, y_train, batch_size=batch_size)
        else:
            data = data_generator
        '''

        if train_strategy is None:
            x_train = self.preprocess_original_imgs(x_train)
            data = self.datagen_rotation.flow(x_train, y_train, batch_size=batch_size)
        else:
            graph = tf.get_default_graph()
            data = DataGenerator(self, model, x_train, y_train, batch_size, train_strategy, graph)

        model.fit_generator(data,
                            steps_per_epoch=len(x_train) / self.batch_size,
                            validation_data=(x_val, y_val),
                            epochs=epochs, verbose=1,
                            callbacks=callbacks_list)
        return model

    def load_model(self, model_id=0, weights_file='gtsrb.hdf5'):
        model = self.def_model(model_id)
        weights_file = os.path.join(self.script_path, weights_file)
        if not os.path.isfile(weights_file):
            logger.fatal("You have not train the model yet, please train model first.")
        model.load_weights(weights_file)
        return model

    def test_dnn_model(self, model=None, print_label="", x_test=None, y_test=None):
        score = model.evaluate(x_test, y_test, verbose=0)
        if print_label != "mute":
            logger.info(print_label + ' - Test loss:' + str(score[0]))
            logger.info(print_label + ' - Test accuracy:' + str(score[1]))
        return score[0], score[1]


if __name__ == '__main__':
    md = FashionMnist()

    _models = [1, "models/try1.hdf5"]

    _model0 = _models
    x_original_train, y_original_train = md.load_original_data("train")
    x_original_test, y_original_test = md.load_original_test_data()

    model=md.train_dnn_model(_model0,
                             x_train=x_original_train, y_train=y_original_train,
                             x_val=x_original_test, y_val=y_original_test)

    # model0 = md.load_model(_model0[0], _model0[1])
    # md.test_dnn_model(model0, "print ", md.preprocess_original_imgs(x_original_test), y_original_test)
