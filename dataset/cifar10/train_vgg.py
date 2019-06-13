"""
Cifar-10 robustness experiment
"""

from __future__ import print_function

import keras
from keras.callbacks import ModelCheckpoint
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from libs.spacial_transformation import *
import tensorflow as tf
import os
from augment.data_generator import DataGenerator
from augment.util import logger
from augment.augmentor import Augmenter
from augment.config import global_config as config
import copy

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras import optimizers
import numpy as np
from keras import regularizers
from keras.initializers import he_normal


class Cifar10Model:
    def __init__(self):

        self.num_classes = 10
        self.IMG_SIZE = 32
        self.batch_size = 32
        self.epoch = 200
        self.depth = 20
        self.input_shape = (self.IMG_SIZE, self.IMG_SIZE, 3)
        self.weight_decay = 0.001

        self.name = "cifar10"

        self.script_path = os.path.dirname(os.path.abspath(__file__))

        (self.x_train, self.y_train), (self.x_test, self.y_test) = cifar10.load_data()

        # define default data generator
        self.datagen_rotation = ImageDataGenerator(
            featurewise_center=False,  # set input mean to 0 over the dataset
            samplewise_center=False,  # set each sample mean to 0
            featurewise_std_normalization=False,  # divide inputs by std of dataset
            samplewise_std_normalization=False,  # divide each input by its std
            zca_whitening=False,  # apply ZCA whitening
            zca_epsilon=1e-06,  # epsilon for ZCA whitening
            rotation_range=0,  # randomly rotate images in the range (deg 0 to 180)
            width_shift_range=0,  # randomly shift images horizontally
            height_shift_range=0,  # randomly shift images vertically
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

    def build_model(self):
        # Build the network of vgg for 10 classes with massive dropout and weight decay as described in the paper.

        # build model
        model = Sequential()
        # Block 1
        model.add(Conv2D(64, (3, 3), padding='same', kernel_regularizer=keras.regularizers.l2(self.weight_decay),
                         kernel_initializer=he_normal(), name='block1_conv1', input_shape=self.input_shape))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Conv2D(64, (3, 3), padding='same', kernel_regularizer=keras.regularizers.l2(self.weight_decay),
                         kernel_initializer=he_normal(), name='block1_conv2'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool'))

        # Block 2
        model.add(Conv2D(128, (3, 3), padding='same', kernel_regularizer=keras.regularizers.l2(self.weight_decay),
                         kernel_initializer=he_normal(), name='block2_conv1'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Conv2D(128, (3, 3), padding='same', kernel_regularizer=keras.regularizers.l2(self.weight_decay),
                         kernel_initializer=he_normal(), name='block2_conv2'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool'))

        # Block 3
        model.add(Conv2D(256, (3, 3), padding='same', kernel_regularizer=keras.regularizers.l2(self.weight_decay),
                         kernel_initializer=he_normal(), name='block3_conv1'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Conv2D(256, (3, 3), padding='same', kernel_regularizer=keras.regularizers.l2(self.weight_decay),
                         kernel_initializer=he_normal(), name='block3_conv2'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Conv2D(256, (3, 3), padding='same', kernel_regularizer=keras.regularizers.l2(self.weight_decay),
                         kernel_initializer=he_normal(), name='block3_conv3'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Conv2D(256, (3, 3), padding='same', kernel_regularizer=keras.regularizers.l2(self.weight_decay),
                         kernel_initializer=he_normal(), name='block3_conv4'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool'))

        # Block 4
        model.add(Conv2D(512, (3, 3), padding='same', kernel_regularizer=keras.regularizers.l2(self.weight_decay),
                         kernel_initializer=he_normal(), name='block4_conv1'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Conv2D(512, (3, 3), padding='same', kernel_regularizer=keras.regularizers.l2(self.weight_decay),
                         kernel_initializer=he_normal(), name='block4_conv2'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Conv2D(512, (3, 3), padding='same', kernel_regularizer=keras.regularizers.l2(self.weight_decay),
                         kernel_initializer=he_normal(), name='block4_conv3'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Conv2D(512, (3, 3), padding='same', kernel_regularizer=keras.regularizers.l2(self.weight_decay),
                         kernel_initializer=he_normal(), name='block4_conv4'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool'))

        # Block 5
        model.add(Conv2D(512, (3, 3), padding='same', kernel_regularizer=keras.regularizers.l2(self.weight_decay),
                         kernel_initializer=he_normal(), name='block5_conv1'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Conv2D(512, (3, 3), padding='same', kernel_regularizer=keras.regularizers.l2(self.weight_decay),
                         kernel_initializer=he_normal(), name='block5_conv2'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Conv2D(512, (3, 3), padding='same', kernel_regularizer=keras.regularizers.l2(self.weight_decay),
                         kernel_initializer=he_normal(), name='block5_conv3'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Conv2D(512, (3, 3), padding='same', kernel_regularizer=keras.regularizers.l2(self.weight_decay),
                         kernel_initializer=he_normal(), name='block5_conv4'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool'))

        # model modification for cifar-10
        model.add(Flatten(name='flatten'))
        model.add(Dense(4096, use_bias=True, kernel_regularizer=keras.regularizers.l2(self.weight_decay),
                        kernel_initializer=he_normal(), name='fc_cifa10'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        model.add(Dense(4096, kernel_regularizer=keras.regularizers.l2(self.weight_decay),
                        kernel_initializer=he_normal(),
                        name='fc2'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        model.add(Dense(10, kernel_regularizer=keras.regularizers.l2(self.weight_decay),
                        kernel_initializer=he_normal(),
                        name='predictions_cifa10'))
        model.add(BatchNormalization())
        model.add(Activation('softmax'))
        return model

    def normalization(self, img_data):
        img_data[:, :, :, 0] = (img_data[:, :, :, 0] - 123.680)
        img_data[:, :, :, 1] = (img_data[:, :, :, 1] - 116.779)
        img_data[:, :, :, 2] = (img_data[:, :, :, 2] - 103.939)

        return img_data

    def cv2_preprocess_img(self, img):
        min_side = min(img.shape[:-1])
        centre = img.shape[0] // 2, img.shape[1] // 2
        img = img[centre[0] - min_side // 2:centre[0] + min_side // 2,
              centre[1] - min_side // 2:centre[1] + min_side // 2]
        img = cv2.resize(img, (self.IMG_SIZE, self.IMG_SIZE))
        return img

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

    def preprocess_original_imgs(self, imgs=None):
        for i in range(len(imgs)):
            imgs[i] = self.cv2_preprocess_img(imgs[i])
        imgs = np.asarray(imgs)
        imgs = imgs.astype('float32')
        imgs = self.normalization(imgs)
        return imgs

    def train_dnn_model(self, _model=None, x_train=None, y_train=None,
                        x_val=None, y_val=None, train_strategy=None):
        """train a dnn model on cifar-10 dataset based on train strategy"""
        learning_rate = 0.1
        lr_decay = 1e-6
        lr_drop = 20

        weights_file = _model[1]
        weights_file = os.path.join(self.script_path, weights_file)

        def lr_scheduler(epoch):
            return learning_rate * (0.5 ** (epoch // lr_drop))
        reduce_lr = keras.callbacks.LearningRateScheduler(lr_scheduler)

        model = self.build_model()

        # optimization details
        sgd = optimizers.SGD(lr=learning_rate, momentum=0.9, nesterov=True)
        model.compile(loss='categorical_crossentropy',
                      optimizer=sgd,
                      metrics=['accuracy'])

        checkpoint = ModelCheckpoint(weights_file, monitor='acc', verbose=1,
                                     save_best_only=False)

        callbacks_list = [checkpoint, reduce_lr]

        x_val = self.preprocess_original_imgs(x_val)
        if train_strategy is None:
            x_train = self.preprocess_original_imgs(x_train)
            self.datagen_rotation.fit(x_train)
            data = self.datagen_rotation.flow(x_train, y_train, batch_size=self.batch_size)
            # model.fit(x_train, y_train,
            #           batch_size=self.batch_size,
            #           epochs=self.epoch,
            #           validation_data=(x_val, y_val),
            #           shuffle=True,
            #           callbacks=callbacks_list)
        else:
            graph = tf.get_default_graph()
            data = DataGenerator(self, model, x_train, y_train, self.batch_size, train_strategy, graph)

        model.fit_generator(data,
                            steps_per_epoch=len(x_train) // self.batch_size,
                            epochs=self.epoch,
                            validation_data=(x_val, y_val),
                            callbacks=callbacks_list,
                            verbose=1)

        return model

    def load_model(self, model_id = 0, weights_file='cifar10.hdf5'):
        model = self.build_model()
        learning_rate = 0.1
        lr_decay = 1e-6
        sgd = optimizers.SGD(lr=learning_rate, decay=lr_decay, momentum=0.9, nesterov=True)
        model.compile(loss='categorical_crossentropy',
                      optimizer=sgd,
                      metrics=['accuracy'])
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


def select_worst2(original_target, model, x_10=None, y=None):
    """Evaluate the loss of each image, and select the worst one based on loss"""
    for i in range(len(x_10)):
        x_10[i] = original_target.preprocess_original_imgs(x_10[i])
    for i in range(10):
        logger.debug("processing round " + str(i))

        set_i = np.array(x_10)[:, i]
        md.test_dnn_model(model, "Train", set_i, y)


def select_worst(original_target, model, x_10=None, y=None, x_train=None, current_indices=None):
    """Evaluate the loss of each image, and select the worst one based on loss"""
    loss = []
    for i in range(len(x_10)):
        x_10[i] = original_target.preprocess_original_imgs(x_10[i])
    for i in range(10):
        logger.debug("processing round " + str(i))
        set_i = np.array(x_10)[:, i]

        y_predict_temp = model.predict(set_i)
        y_true1 = np.array(y, dtype='float32')

        y_true1 = tf.convert_to_tensor(y_true1)
        y_pred1 = tf.convert_to_tensor(y_predict_temp)

        loss1 = keras.losses.categorical_crossentropy(y_true1, y_pred1)
        loss1 = keras.backend.get_value(loss1)
        loss.append(loss1)
    logger.debug("loss generation done!!!")
    y_argmax = np.argmax(loss, axis=0)
    y_max = np.max(loss, axis=0)

    logger.debug("length of y_argmax: " + str(len(y_argmax)))
    if config.enable_optimize:
        temp_indices = copy.deepcopy(current_indices)
    for j in range(len(y_argmax)):
        if config.enable_optimize:
            x_index = temp_indices[j]
            index = y_argmax[j]
            x_train[x_index] = x_10[j][index]  # update x_train (not good design)
            if y_max[j] < 1e-3:
                current_indices.remove(x_index)
        else:
            index = y_argmax[j]
            x_train[j] = x_10[j][index]  # update x_train (not good design)
    return x_train

if __name__ == '__main__':
    md = Cifar10Model()
    _model0 = [0, "models/cifar10replace_worst_of_10_model_False.hdf5"]
    x_train_origin, y_train_origin = md.load_original_data('train')
    x_val_origin, y_val_origin = md.load_original_data('val')

    current_indices = range(len(x_train_origin))

    au = Augmenter()
    x_train_origin_temp = copy.copy(x_train_origin)
    x_10, y_train = au.worst_of_10(x_train_origin_temp, y_train_origin)

    x_train = md.preprocess_original_imgs(x_train_origin)
    model0 = md.load_model(0, _model0[1])
    select_worst(md, model0, x_10, y_train, x_train, current_indices)

    md.test_dnn_model(model0, "Train", x_train, y_train_origin)

    # md.test_dnn_model(model0, "Train", md.preprocess_original_imgs(x_train_origin), y_train_origin)
    # md.test_dnn_model(model0, "Test", md.preprocess_original_imgs(x_val_origin), y_val_origin)

