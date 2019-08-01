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
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Activation, GlobalAveragePooling2D
from keras import backend as k
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
from skimage import color, exposure, transform
import tensorflow as tf
import cv2
from augment.data_generator import DataGenerator
from augment.util import logger
from augment.config import ExperimentalConfig
from keras.regularizers import l1, l2
from keras.applications.mobilenet import MobileNet
from keras.optimizers import Nadam

import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

import warnings

warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', category=DeprecationWarning)

import pandas as pd

pd.options.display.max_columns = 100

from PIL import Image
from scipy import ndimage

from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.optimizers import SGD, Adam, RMSprop, Adagrad, Nadam, Adadelta, Adamax
from keras.utils import to_categorical


def prepare_date():
    images1 = []
    targets1 = []

    path1 = "./kvasir-dataset/dyed-lifted-polyps"
    path2 = "./kvasir-dataset/dyed-resection-margins"
    path3 = "./kvasir-dataset/esophagitis"
    path4 = "./kvasir-dataset/normal-cecum"
    path5 = "./kvasir-dataset/normal-pylorus"
    path6 = "./kvasir-dataset/normal-z-line"
    path7 = "./kvasir-dataset/polyps"
    path8 = "./kvasir-dataset/ulcerative-colitis"

    for i in glob.glob(os.path.join(path1, '*jpg')):
        img = cv2.imread(i)
        # img  = cv2.cvtColor(img , cv2.COLOR_BGR2GRAY) # convert to grayscale
        img = cv2.resize(img, (50, 50))  # resize
        images1.append(np.array(img))
        targets1.append(0)

    for j in glob.glob(os.path.join(path2, '*jpg')):
        img = cv2.imread(j)
        # img  = cv2.cvtColor(img , cv2.COLOR_BGR2GRAY) # convert to grayscale
        img = cv2.resize(img, (50, 50))  # resize
        images1.append(np.array(img))
        targets1.append(1)

    for k in glob.glob(os.path.join(path3, '*jpg')):
        img = cv2.imread(k)
        # img  = cv2.cvtColor(img , cv2.COLOR_BGR2GRAY) # convert to grayscale
        img = cv2.resize(img, (50, 50))  # resize
        images1.append(np.array(img))
        targets1.append(2)

    for l in glob.glob(os.path.join(path4, '*jpg')):
        img = cv2.imread(l)
        # img  = cv2.cvtColor(img , cv2.COLOR_BGR2GRAY) # convert to grayscale
        img = cv2.resize(img, (50, 50))  # resize
        images1.append(np.array(img))
        targets1.append(3)

    for m in glob.glob(os.path.join(path5, '*jpg')):
        img = cv2.imread(m)
        # img  = cv2.cvtColor(img , cv2.COLOR_BGR2GRAY) # convert to grayscale
        img = cv2.resize(img, (50, 50))  # resize
        images1.append(np.array(img))
        targets1.append(4)

    for n in glob.glob(os.path.join(path6, '*jpg')):
        img = cv2.imread(n)
        # img  = cv2.cvtColor(img , cv2.COLOR_BGR2GRAY) # convert to grayscale
        img = cv2.resize(img, (50, 50))  # resize
        images1.append(np.array(img))
        targets1.append(5)

    for o in glob.glob(os.path.join(path7, '*jpg')):
        img = cv2.imread(o)
        # img  = cv2.cvtColor(img , cv2.COLOR_BGR2GRAY) # convert to grayscale
        img = cv2.resize(img, (50, 50))  # resize
        images1.append(np.array(img))
        targets1.append(6)

    for p in glob.glob(os.path.join(path8, '*jpg')):
        img = cv2.imread(p)
        # img  = cv2.cvtColor(img , cv2.COLOR_BGR2GRAY) # convert to grayscale
        img = cv2.resize(img, (50, 50))  # resize
        images1.append(np.array(img))
        targets1.append(7)

    endoscope_images1, endoscope_labels1 = np.array(images1), np.array(targets1)
    np.save("endoscope_images1", endoscope_images1)
    np.save("endoscope_labels1", endoscope_labels1)


class KvasirModel:
    """ source dir is the relative path of gtsrb data set' """

    def __init__(self, source_dir="kvasir-dataset", start_point=0, epoch=50):

        self.config = ExperimentalConfig.gen_config()
        # Config related to images in the gtsrb dataset
        self.start_point = start_point

        self.num_classes = 8
        self.IMG_SIZE = 50
        self.epoch = epoch
        self.name = "kvasir"
        self.batch_size = 128
        self.input_shape = (self.IMG_SIZE, self.IMG_SIZE, 3)
        self.script_path = os.path.dirname(os.path.abspath(__file__))
        self.root_dir = os.path.join(self.script_path, source_dir)
        if not os.path.isdir(self.root_dir):
            logger.fatal("Please download the dataset first.")
        #prepare_date()

        __image_path = os.path.join(self.script_path, "endoscope_images1.npy")
        __label_path = os.path.join(self.script_path, "endoscope_labels1.npy")
        the_endoscope_images1, the_endoscope_labels1 = np.load(__image_path), \
                                                       np.load(__label_path)

        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(the_endoscope_images1,
                                                                                the_endoscope_labels1,
                                                                                test_size=0.2)

        # define default data generator
        self.datagen_rotation = ImageDataGenerator(
            featurewise_center=False,  # set input mean to 0 over the dataset
            samplewise_center=False,  # set each sample mean to 0
            featurewise_std_normalization=False,  # divide inputs by std of dataset
            samplewise_std_normalization=False,  # divide each input by its std
            zca_whitening=False,  # apply ZCA whitening
            zca_epsilon=1e-06,  # epsilon for ZCA whitening
            rotation_range=30,  # randomly rotate images in the range (deg 0 to 180)
            width_shift_range=0.1,  # randomly shift images horizontally
            height_shift_range=0.1,  # randomly shift images vertically
            shear_range=0.2,  # set range for random shear
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
        if model_id == 0:
            cnndeepmodel = Sequential()
            # first convolution layer
            cnndeepmodel.add(
                Conv2D(filters=16, kernel_size=2, padding="same", activation="relu", input_shape=self.input_shape))
            cnndeepmodel.add(BatchNormalization())
            cnndeepmodel.add(MaxPooling2D(pool_size=2))
            cnndeepmodel.add(Dropout(0.25))

            # second convolution  layer
            cnndeepmodel.add(Conv2D(filters=32, kernel_size=2, padding="same", activation="relu"))
            cnndeepmodel.add(BatchNormalization())
            cnndeepmodel.add(MaxPooling2D(pool_size=2))
            cnndeepmodel.add(Dropout(0.5))

            # Third convolution  layer
            cnndeepmodel.add(Conv2D(64, kernel_size=2, padding="same", activation='relu'))
            cnndeepmodel.add(BatchNormalization())
            cnndeepmodel.add(MaxPooling2D(pool_size=2))
            cnndeepmodel.add(Dropout(0.5))

            # first Fully connected layer
            cnndeepmodel.add(Flatten())
            cnndeepmodel.add(Dense(256, kernel_regularizer=l2(0.001)))
            cnndeepmodel.add(BatchNormalization())
            cnndeepmodel.add(Activation('relu'))
            cnndeepmodel.add(Dropout(0.5))

            # Final Fully connected layer
            cnndeepmodel.add(Dense(8))
            cnndeepmodel.add(Activation('softmax'))

            cnndeepmodel.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(),
                                 metrics=['accuracy'])

            cnndeepmodel.summary()
            return cnndeepmodel
        elif model_id == 1:
            base_model = MobileNet(
                input_shape=self.input_shape,
                weights=None,
                include_top=False)

            # apply global pooling to output from base model
            x = base_model.output
            x = GlobalAveragePooling2D()(x)

            # add classification layer to model (this is dataset specific
            # as number of units must equal number of classes)
            predictions = Dense(
                units=8,
                activation='softmax',
                name='predictions')(x)

            learning_rate = 0.01
            beta_1 = 0.9
            beta_2 = 0.999
            epsilon = None
            schedule_decay = 0.004

            optimizer = Nadam(
                lr=learning_rate,
                beta_1=beta_1,
                beta_2=beta_2,
                epsilon=epsilon,
                schedule_decay=schedule_decay)

            # combine layers to get complete model
            model = Model(base_model.input, predictions)
            model.compile(
                loss='categorical_crossentropy',
                optimizer=optimizer,
                metrics=['accuracy'])

            return model

        else:
            raise Exception("unsupported model")

    def load_original_data(self, data):
        Y_train = keras.utils.to_categorical(self.Y_train, self.num_classes)
        return list(self.X_train), Y_train

    def load_original_test_data(self):
        Y_test = keras.utils.to_categorical(self.Y_test, self.num_classes)
        return list(self.X_test), Y_test

    def preprocess_original_imgs(self, imgs=None):
        for i in range(len(imgs)):
            imgs[i] = transform.resize(imgs[i], (self.IMG_SIZE, self.IMG_SIZE))
        imgs = np.asarray(imgs)
        imgs = np.array(imgs, dtype='float32')/255
        return imgs

    def train_dnn_model(self, _model=[1, "models/kvasir.hdf5"], x_train=None, y_train=None,
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
        # callbacks_list = [checkpoint, lr_reducer, lr_scheduler]

        x_val = self.preprocess_original_imgs(x_val)

        if train_strategy is None:
            x_train = self.preprocess_original_imgs(x_train)
            data = self.datagen_rotation.flow(x_train, y_train, batch_size=batch_size)
        else:
            graph = tf.get_default_graph()
            data = DataGenerator(self, model, x_train, y_train, batch_size, train_strategy, graph)

        model.fit_generator(data,
                            validation_data=(x_val, y_val),
                            epochs=epochs, verbose=1,
                            steps_per_epoch=len(x_train) / self.batch_size,
                            callbacks=callbacks_list)

        return model

    def load_model(self, model_id=0, weights_file='models/kvasir.hdf5'):
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
    md = KvasirModel()

    _models = [(0, "models/kvasir0.hdf5"), (1, "models/kvasir0.hdf5")]

    _model = _models[1]
    x_train, y_train = md.load_original_data()
    x_val, y_val = md.load_original_test_data()

    model = md.train_dnn_model(_model,
                               x_train=x_train, y_train=y_train,
                               x_val=x_val, y_val=y_val)

