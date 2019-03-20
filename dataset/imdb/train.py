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
from keras.models import Model, Sequential
from keras.layers import Dense, Dropout, Flatten, Input, GlobalAveragePooling2D,AveragePooling2D
from keras import backend as k
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
import tensorflow as tf
import cv2
from deepaugment.data_generator import DataGenerator
from deepaugment.util import logger
from deepaugment.config import ExperimentalConfig
from keras.applications.vgg16 import VGG16
import h5py


def cv2_preprocess_img(img, img_size):
    img = cv2.resize(img, (img_size, img_size))
    # img = img.reshape(self.IMG_SIZE, self.IMG_SIZE, 1)
    return img


class IMDBModel:
    """ source dir is the relative path of gtsrb data set' """

    def __init__(self, source_dir=None, start_point=0, epoch=30):

        self.config = ExperimentalConfig.gen_config()
        # Config related to images in the gtsrb dataset
        self.start_point = start_point

        self.num_classes = 2
        self.IMG_SIZE = 64
        self.epoch = epoch
        self.name = "imdb"
        self.batch_size = 64
        self.input_shape = (self.IMG_SIZE, self.IMG_SIZE, 3)
        self.script_path = os.path.dirname(os.path.abspath(__file__))
        self.root_dir = os.path.join(self.script_path, source_dir)

        (self.x_train, self.y_train), (self.x_test, self.y_test) = self.load_imdb_data(self.root_dir)

        if not os.path.isdir(self.root_dir):
            logger.fatal("Please download the dataset first.")

        # define default data generator
        self.datagen_rotation = ImageDataGenerator(
            featurewise_center=False,  # set input mean to 0 over the dataset
            samplewise_center=False,  # set each sample mean to 0
            featurewise_std_normalization=False,  # divide inputs by std of dataset
            samplewise_std_normalization=False,  # divide each input by its std
            zca_whitening=False,  # apply ZCA whitening
            zca_epsilon=1e-06,  # epsilon for ZCA whitening
            rotation_range=0,  # randomly rotate images in the range (deg 0 to 180)
            width_shift_range=0.1,  # randomly shift images horizontally
            height_shift_range=0.1,  # randomly shift images vertically
            shear_range=0,  # set range for random shear
            zoom_range=0.,  # set range for random zoom
            channel_shift_range=0.,  # set range for random channel shifts
            fill_mode='nearest',  # set mode for filling points outside the input boundaries
            cval=0.,  # value used for fill_mode = "constant"
            horizontal_flip=True,  # randomly flip images
            vertical_flip=False,  # randomly flip images
            rescale=None,  # set rescaling factor (applied before any other transformation)
            preprocessing_function=None,  # set function that will be applied on each input
            data_format=None,  # image data format, either "channels_first" or
            # "channels_last"
            validation_split=0.0  # fraction of images reserved for validation
            # (strictly between 0 and 1)
        )

    def def_model(self, model_id=0):
        # self.input_shape = (image_size, image_size, 3)
        if model_id == 0:
            input_tensor = Input(shape=self.input_shape)
            base_model = VGG16(weights='imagenet', include_top=False, input_tensor=input_tensor)
            x = base_model.output
            x = GlobalAveragePooling2D()(x)
            x = Dense(1024, activation='relu')(x)
            predictions = Dense(self.num_classes, activation='softmax')(x)
            model = Model(inputs=base_model.input, outputs=predictions)
            for layer in base_model.layers[:15]:
                layer.trainable = False
            model.compile(optimizer=keras.optimizers.SGD(lr=0.0001, momentum=0.9, decay=1e-6),
                          loss='categorical_crossentropy',
                          metrics=['accuracy'])
            return model
        else:
            raise Exception("unsupported model")

    def load_original_data(self, data_type='train'):
        if data_type == 'train':
            return list(self.x_train), self.y_train
        else:  # without validation set
            return list(self.x_test), self.y_test

    def load_original_test_data(self):
        return list(self.x_test), self.y_test

    # load svhn data from the specified folder
    def load_imdb_data(self, path):
        f1 = h5py.File(path+'/train.h5', 'r')
        x_train = f1["X"][:]
        y_train = f1["Y"][:]
        del f1

        f2 = h5py.File(path+'/test.h5', 'r')
        x_test = f2["X"][:]
        y_test = f2["Y"][:]
        del f2

        # y_train = keras.utils.to_categorical(y_train, 2)
        # y_test = keras.utils.to_categorical(y_test, 2)

        return (x_train, y_train), (x_test, y_test)

    def preprocess_original_imgs(self, imgs=None):
        for i in range(len(imgs)):
            imgs[i] = cv2_preprocess_img(imgs[i], self.IMG_SIZE)
        imgs = np.array(imgs, dtype='float32')
        return imgs

    def train_dnn_model(self, _model=[0, "models/gtsrb"], x_train=None, y_train=None,
                        x_val=None, y_val=None, train_strategy=None):
        """
        train a dnn model on gstrb dataset based on train strategy
        """
        # training config
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

        if train_strategy is None:
            x_train = self.preprocess_original_imgs(x_train)
            data = self.datagen_rotation.flow(x_train, y_train, batch_size=batch_size)
        else:
            graph = tf.get_default_graph()
            data = DataGenerator(self, model, x_train, y_train, batch_size, train_strategy, graph)

        # if model_id == 0:
        model.fit_generator(data,
                            steps_per_epoch=len(self.x_train)/self.batch_size,
                            validation_data=(x_val, y_val),
                            epochs=epochs, verbose=1,
                            callbacks=callbacks_list)
        # elif model_id == 1:
        #      for layer in model.layers:
        #          layer.trainable = True
        #      model.load_weights(weights_file)
        #    model.fit_generator(data,
        #                         validation_data=(x_val, y_val),
        #                         epochs=epochs, verbose=1,
        #                         shuffle=False,
        #                         callbacks=callbacks_list)

        # score = model.evaluate(x_train, y_train, verbose=0)
        # print('Train loss:', score[0])
        # print('Train accuracy:', score[1])

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
    md = IMDBModel(source_dir='dataset')

    _models = [[0, "models/try1.hdf5"]]
    # 0.9616785431229115
    # 0.9038004751065754
    # 0.971496437026316
    _model0 = _models[0]
    x_train, y_train = md.load_original_data('train')
    x_val, y_val = md.load_original_data('val')
    # x_original_test, y_original_test = md.load_original_test_data()

    model = md.train_dnn_model(_model0,
                               x_train=md.preprocess_original_imgs(x_train), y_train=y_train,
                               x_val=md.preprocess_original_imgs(x_val), y_val=y_val)

