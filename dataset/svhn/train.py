from __future__ import print_function

import os
import sys
import numpy as np
import cv2
import dateutil.tz
import h5py
import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten, Activation
from keras.layers.normalization import BatchNormalization
from keras.preprocessing import image
from deepaugment.data_generator import DataGenerator
from deepaugment.util import logger
from deepaugment.config import ExperimentalConfig
import tensorflow as tf
from keras import backend as k
from keras.preprocessing.image import ImageDataGenerator


class SVHN:

    def __init__(self, data_path="data", start_point=0, epoch=30):
        self.script_path = os.path.dirname(os.path.abspath(__file__))
        root_dir = os.path.join(self.script_path, data_path)
        if not os.path.isdir(root_dir):
            logger.fatal("Please download the dataset first.")
            exit(1)

        self.config = ExperimentalConfig.gen_config()
        # Config related to images in the gtsrb dataset
        self.start_point = start_point
        self.epoch = epoch

        # load data
        (self.x_train, self.y_train), (self.x_test, self.y_test) = self.load_svhn_data(root_dir)

        self.num_classes = 10
        self.name = "svhn"
        self.batch_size = 128
        self.image_size = 32
        self.input_shape = (self.image_size, self.image_size, 3)
        print("running 1.5")

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

    def load_original_data(self, data_type='train'):
        if data_type == 'train':
            return list(self.x_train), self.y_train
        else:  # without validation set
            return list(self.x_test), self.y_test

    def load_original_test_data(self):
        return list(self.x_test), self.y_test

    def preprocess_original_imgs(self, x):
        for i in range(len(x)):
            x[i] = cv2.resize(x[i], (self.image_size, self.image_size))
        return np.array(x)

    # load svhn data from the specified folder
    def load_svhn_data(self, path):
        f1 = h5py.File(path+'/train.hdf5', 'r')
        x_train = f1["X"][:]
        y_train = f1["Y"][:]
        del f1

        f2 = h5py.File(path+'/test.hdf5', 'r')
        x_test = f2["X"][:]
        y_test = f2["Y"][:]
        del f2

        y_train = keras.utils.to_categorical(y_train, 10)
        y_test = keras.utils.to_categorical(y_test, 10)

        return (x_train, y_train), (x_test, y_test)

    # build the classification model
    def def_model(self):
        input_shape = self.input_shape
        model = Sequential()
        model.add(Conv2D(32, kernel_size=3, input_shape=input_shape, padding="same"))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Conv2D(32, 3, padding="same"))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=2))
        model.add(Dropout(0.3))

        model.add(Conv2D(64, 3))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Conv2D(64, 3, padding="same"))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=2))
        model.add(Dropout(0.3))

        model.add(Conv2D(128, 3))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Conv2D(128, 3, padding="same"))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=2))
        model.add(Dropout(0.3))

        model.add(Flatten())
        model.add(Dense(512, activation='relu'))
        model.add(Dropout(0.3))
        model.add(Dense(10, activation='softmax'))

        lr = 1e-3
        model.compile(loss=keras.losses.categorical_crossentropy,
                      optimizer=keras.optimizers.Adam(lr=lr),
                      metrics=['accuracy'])
        return model

    def train_dnn_model(self, _model=[0, "models/svhn.hdf5"], x_train=None, y_train=None,
                        x_val=None, y_val=None, train_strategy=None):
        """
        train a dnn model on svhn dataset based on train strategy
        """

        tf.keras.backend.clear_session()

        epochs = self.epoch
        batch_size = self.batch_size

        k.set_image_data_format('channels_last')

        # model_id = _model[0]
        weights_file = _model[1]
        weights_file = os.path.join(self.script_path, weights_file)

        model = self.def_model()
        if self.start_point > 0:
            model.load_weights(weights_file)

        checkpoint = keras.callbacks.ModelCheckpoint(weights_file, monitor='acc', verbose=1,
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

    def load_model(self, model_id=0, weights_file='svhn.hdf5'):
        model = self.def_model()
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
    md = SVHN()

    _models = [1, "models/try1.hdf5"]

    _model0 = _models
    x_original_train, y_original_train = md.load_original_data("train")
    x_original_test, y_original_test = md.load_original_test_data()
    md.train_dnn_model(_model0,
                       x_train=x_original_train, y_train=y_original_train,
                       x_val=x_original_test, y_val=y_original_test)
