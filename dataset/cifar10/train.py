"""
Cifar-10 robustness experiment
"""

from __future__ import print_function

import keras
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, ReduceLROnPlateau
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import os
import copy
from deepaugment.data_generator import DataGenerator
from deepaugment.util import logger
import cv2
import numpy as np
from keras.layers import Conv2D, Dense, Input, add, Activation, GlobalAveragePooling2D, AveragePooling2D
from keras.models import Model
from keras import optimizers, regularizers
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras.initializers import he_normal
from keras.applications.vgg19 import VGG19
from keras.regularizers import l2
import keras.backend as k

IN_FILTERS = 16


def scheduler(epoch):
    if epoch < 60:
        return 0.1
    if epoch < 120:
        return 0.02
    if epoch < 160:
        return 0.004
    return 0.0008


def residual_network(img_input, classes_num=10, stack_n=5, weight_decay=1e-4):
    def residual_block(x, o_filters, increase=False):
        stride = (1, 1)
        if increase:
            stride = (2, 2)

        o1 = Activation('relu')(BatchNormalization(momentum=0.9, epsilon=1e-5)(x))
        conv_1 = Conv2D(o_filters, kernel_size=(3, 3), strides=stride, padding='same',
                        kernel_initializer="he_normal",
                        kernel_regularizer=regularizers.l2(weight_decay))(o1)
        o2 = Activation('relu')(BatchNormalization(momentum=0.9, epsilon=1e-5)(conv_1))
        conv_2 = Conv2D(o_filters, kernel_size=(3, 3), strides=(1, 1), padding='same',
                        kernel_initializer="he_normal",
                        kernel_regularizer=regularizers.l2(weight_decay))(o2)
        if increase:
            projection = Conv2D(o_filters, kernel_size=(1, 1), strides=(2, 2), padding='same',
                                kernel_initializer="he_normal",
                                kernel_regularizer=regularizers.l2(weight_decay))(o1)
            block = add([conv_2, projection])
        else:
            block = add([conv_2, x])
        return block

    # build model ( total layers = stack_n * 3 * 2 + 2 )
    # stack_n = 5 by default, total layers = 32
    # input: 32x32x3 output: 32x32x16
    x = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same',
               kernel_initializer="he_normal",
               kernel_regularizer=regularizers.l2(weight_decay))(img_input)

    # input: 32x32x16 output: 32x32x16
    for _ in range(stack_n):
        x = residual_block(x, 16, False)

    # input: 32x32x16 output: 16x16x32
    x = residual_block(x, 32, True)
    for _ in range(1, stack_n):
        x = residual_block(x, 32, False)

    # input: 16x16x32 output: 8x8x64
    x = residual_block(x, 64, True)
    for _ in range(1, stack_n):
        x = residual_block(x, 64, False)

    x = BatchNormalization(momentum=0.9, epsilon=1e-5)(x)
    x = Activation('relu')(x)
    x = GlobalAveragePooling2D()(x)

    # input: 64 output: 10
    x = Dense(classes_num, activation='softmax', kernel_initializer="he_normal",
              kernel_regularizer=regularizers.l2(weight_decay))(x)

    model = Model(img_input, x)

    sgd = optimizers.SGD(lr=.1, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    return model


def vgg_model(input_shape, num_classes):
    # Build the network of vgg for 10 classes with massive dropout and weight decay as described in the paper.
    """
    model with batch normalization
    https://github.com/xitizzz/Traffic-Sign-Recognition-using-Deep-Neural-Network
    """
    model = Sequential()
    BatchNormalization(epsilon=1e-06, momentum=0.99, weights=None)

    # Block 1 Convolution layer 1,2  Normalization layer 3
    model.add(Conv2D(32, (3, 3), padding='same', input_shape=input_shape, activation='relu'))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.2))

    # Block 1 Convolution layer 4,5  Normalization layer 6
    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.2))

    # Block 1 Convolution layer 7,8  Normalization layer 9
    model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.2))

    # Block 5 Fully-connected layer 10, Normalization layer 11,  Output layer 12
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(BatchNormalization())
    model.add(Dense(num_classes, activation='softmax'))

    opt = keras.optimizers.SGD(lr=0.0001, decay=1e-6)

    model.compile(loss='categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])
    return model


def wide_residual_network(input_shape, num_class):

    weight_decay = 0.0005
    depth = 28
    k = 10
    n_filters = [16, 16 * k, 32 * k, 64 * k]
    n_stack = (depth - 4) // 6

    def conv3x3(x, filters):
        return Conv2D(filters=filters, kernel_size=(3, 3), strides=(1, 1), padding='same',
                      kernel_initializer='he_normal',
                      kernel_regularizer=l2(weight_decay),
                      use_bias=False)(x)

    def bn_relu(x):
        x = BatchNormalization(momentum=0.9, epsilon=1e-5)(x)
        x = Activation('relu')(x)
        return x

    def residual_block(x, out_filters, increase=False):
        global IN_FILTERS
        stride = (1, 1)
        if increase:
            stride = (2, 2)

        o1 = bn_relu(x)

        conv_1 = Conv2D(out_filters,
                        kernel_size=(3, 3), strides=stride, padding='same',
                        kernel_initializer='he_normal',
                        kernel_regularizer=l2(weight_decay),
                        use_bias=False)(o1)

        o2 = bn_relu(conv_1)

        conv_2 = Conv2D(out_filters,
                        kernel_size=(3, 3), strides=(1, 1), padding='same',
                        kernel_initializer='he_normal',
                        kernel_regularizer=l2(weight_decay),
                        use_bias=False)(o2)
        if increase or IN_FILTERS != out_filters:
            proj = Conv2D(out_filters,
                          kernel_size=(1, 1), strides=stride, padding='same',
                          kernel_initializer='he_normal',
                          kernel_regularizer=l2(weight_decay),
                          use_bias=False)(o1)
            block = add([conv_2, proj])
        else:
            block = add([conv_2, x])
        return block

    def wide_residual_layer(x, out_filters, increase=False):
        global IN_FILTERS
        x = residual_block(x, out_filters, increase)
        IN_FILTERS = out_filters
        for _ in range(1, int(n_stack)):
            x = residual_block(x, out_filters)
        return x

    img_input = Input(shape=input_shape)
    x = conv3x3(img_input, n_filters[0])
    x = wide_residual_layer(x, n_filters[1])
    x = wide_residual_layer(x, n_filters[2], increase=True)
    x = wide_residual_layer(x, n_filters[3], increase=True)
    x = BatchNormalization(momentum=0.9, epsilon=1e-5)(x)
    x = Activation('relu')(x)
    x = AveragePooling2D((8, 8))(x)
    x = Flatten()(x)
    x = Dense(num_class,
              activation='softmax',
              kernel_initializer='he_normal',
              kernel_regularizer=l2(weight_decay),
              use_bias=False)(x)

    resnet = Model(img_input, x)
    # set optimizer
    sgd = optimizers.SGD(lr=.1, momentum=0.9, nesterov=True)
    resnet.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

    return resnet


def vgg19(input_tensor, num_class):

    base_model = VGG19(include_top=False, input_tensor=input_tensor)

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu')(x)
    predictions = Dense(num_class, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)

    model.compile(optimizer=optimizers.SGD(lr=0.0001, momentum=0.9),
                  loss='categorical_crossentropy', metrics=['accuracy'])
    return model


class Cifar10Model:
    def __init__(self, start_point=0, epoch=200):

        self.start_point = start_point
        self.num_classes = 10
        self.IMG_SIZE = 32
        self.batch_size = 512
        self.epoch = epoch
        self.input_shape = (self.IMG_SIZE, self.IMG_SIZE, 3)

        self.stack_n = 5
        # self.layers = 6 * self.stack_n + 2
        self.weight_decay = 1e-4

        self.min_value = 1e-8
        self.max_value = 1 - 1e-8

        self.name = "cifar10"

        self.script_path = os.path.dirname(os.path.abspath(__file__))

        (self.x_train, self.y_train), (self.x_test, self.y_test) = cifar10.load_data()

        x_train = copy.deepcopy(self.x_train)
        x_train = x_train.astype('float32') / 255
        self.train_mean = np.mean(x_train, axis=0)
        self.train_std = np.std(x_train, axis=(0, 1, 2))
        del x_train

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

    def normalization2(self, img_data):
        return (img_data - self.train_mean) / self.train_std

    def normalization(self, img_data):
        mean = [125.307, 122.95, 113.865]
        std = [62.9932, 62.0887, 66.7048]
        for i in range(3):
            img_data[:, :, :, i] = (img_data[:, :, :, i] - mean[i]) / std[i]
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

    def scheduler(self, epoch):
        epoch += self.start_point
        if epoch < 81:
            return 0.1
        if epoch < 162:
            return 0.01
        return 0.001

    def train_dnn_model(self, _model=None, x_train=None, y_train=None,
                        x_val=None, y_val=None, train_strategy=None):
        """train a dnn model on cifar-10 dataset based on train strategy"""
        # k.set_image_data_format('channels_last')
        model_id = _model[0]
        weights_file = _model[1]
        weights_file = os.path.join(self.script_path, weights_file)

        checkpoint = ModelCheckpoint(weights_file, monitor='acc', verbose=1,
                                     save_best_only=False)
        callbacks_list = [LearningRateScheduler(self.scheduler),
                          checkpoint]

        img_input = Input(shape=self.input_shape)
        if model_id == 0:
            model = residual_network(img_input, self.num_classes, 5, self.weight_decay)
        elif model_id == 1:
            model = residual_network(img_input, self.num_classes, 8, self.weight_decay)
        elif model_id == 2:
            self.batch_size = 64
            model = wide_residual_network(self.input_shape, self.num_classes)
            change_lr = LearningRateScheduler(scheduler)
            callbacks_list.append(change_lr)
        elif model_id == 3:
            model = residual_network(img_input, self.num_classes, 3, self.weight_decay)
        else:
            logger.fatal("unsupported model id")
            exit(2)

        if self.start_point > 0:
            model.load_weights(weights_file)

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
                            steps_per_epoch=len(x_train)/self.batch_size,
                            validation_data=(x_val, y_val),
                            epochs=self.epoch, verbose=1, #workers=4,
                            callbacks=callbacks_list)
        return model

    def load_model(self, model_id=0, weights_file='cifar10.hdf5'):
        if model_id == 0:
            img_input = Input(shape=self.input_shape)
            model = residual_network(img_input, self.num_classes, 5, self.weight_decay)
        elif model_id == 1:
            img_input = Input(shape=self.input_shape)
            model = residual_network(img_input, self.num_classes, 8, self.weight_decay)
        elif model_id == 2:
            self.batch_size = 64
            model = wide_residual_network(self.input_shape, self.num_classes)
        elif model_id == 3:
            img_input = Input(shape=self.input_shape)
            model = residual_network(img_input, self.num_classes, 3, self.weight_decay)
        else:
            logger.fatal("unsupported model id")
            exit(2)

        # sgd = optimizers.SGD(lr=.1, momentum=0.9, nesterov=True)
        # model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

        weights_file = os.path.join(self.script_path, weights_file)
        if not os.path.isfile(weights_file):
            logger.fatal("You have not trained the model yet, please train model first.")
        model.load_weights(weights_file)
        return model

    def test_dnn_model(self, model=None, print_label="", x_test=None, y_test=None):
        score = model.evaluate(x_test, y_test, verbose=0)
        if print_label != "mute":
            logger.info(print_label + ' - Test loss:' + str(score[0]))
            logger.info(print_label + ' - Test accuracy:' + str(score[1]))
        return score[0], score[1]
