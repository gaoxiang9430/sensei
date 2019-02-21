"""
Cifar-10 robustness experiment
"""

from __future__ import print_function

import keras
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, ReduceLROnPlateau
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from libs.spacial_transformation import *
import tensorflow as tf
from resnet import resnet_v1, lr_schedule
import os
import copy
from deepaugment.data_generator import DataGenerator
from deepaugment.util import logger


class Cifar10Model:

    def __init__(self):
        
        self.num_classes = 10       
        self.IMG_SIZE = 32
        self.batch_size = 128
        self.epoch = 200
        self.depth = 20
        self.input_shape = (self.IMG_SIZE, self.IMG_SIZE, 3)

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
            featurewise_center=False,             # set input mean to 0 over the dataset
            samplewise_center=False,              # set each sample mean to 0
            featurewise_std_normalization=False,  # divide inputs by std of dataset
            samplewise_std_normalization=False,   # divide each input by its std
            zca_whitening=False,                  # apply ZCA whitening
            zca_epsilon=1e-06,                    # epsilon for ZCA whitening
            rotation_range=0,                     # randomly rotate images in the range (deg 0 to 180)
            width_shift_range=0,                  # randomly shift images horizontally
            height_shift_range=0,                 # randomly shift images vertically
            shear_range=0,                        # set range for random shear
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

    def normalization2(self, img_data):
        return (img_data - self.train_mean) / self.train_std

    def normalization(self, img_data):
        return img_data / 255 - self.train_mean

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
        else:                              # without validation set
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
        # k.set_image_data_format('channels_last')

        # model_id = _model[0]
        weights_file = _model[1]
        weights_file = os.path.join(self.script_path, weights_file)

        model = resnet_v1(input_shape=self.input_shape, depth=self.depth)

        def categorical_crossentropy_wrapper(y_true, y_pred):
            y_pred = keras.backend.clip(y_pred, self.min_value, self.max_value)
            return keras.losses.categorical_crossentropy(y_true, y_pred)
        model.compile(loss=categorical_crossentropy_wrapper,
                      optimizer=Adam(lr=lr_schedule(0)),
                      metrics=['accuracy'])

        lr_scheduler = LearningRateScheduler(lr_schedule)
        lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
                                       cooldown=0,
                                       patience=5,
                                       min_lr=0.5e-6)

        checkpoint = ModelCheckpoint(weights_file, monitor='acc', verbose=1,
                                     save_best_only=False)

        callbacks_list = [checkpoint, lr_reducer, lr_scheduler]

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
                            validation_data=(x_val, y_val),
                            epochs=self.epoch, verbose=1, workers=4,
                            callbacks=callbacks_list)
        return model

    def load_model(self, model_id=0, weights_file='cifar10.hdf5'):
        model = resnet_v1(input_shape=self.input_shape, depth=self.depth)
        model.compile(loss='categorical_crossentropy',
                      optimizer=Adam(lr=lr_schedule(self.epoch)),
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

if __name__ == '__main__':
    md = Cifar10Model()
    _model0 = [0, "models/cifar10.resnet56v1_3.hdf5"]
    x_train_origin, y_train_origin = md.load_original_data('train')
    x_val_origin, y_val_origin = md.load_original_data('val')

    model0 = md.train_dnn_model(_model=_model0,
                                x_train=x_train_origin, y_train=y_train_origin,
                                x_val=x_val_origin, y_val=y_val_origin)
