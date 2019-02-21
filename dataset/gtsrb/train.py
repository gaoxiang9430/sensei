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
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as k
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
from skimage import color, exposure, transform
import tensorflow as tf
import cv2
from deepaugment.data_generator import DataGenerator
from deepaugment.util import logger
from deepaugment.config import global_config as config


def preprocess_img(self, img):
    # Histogram normalization in v channel
    hsv = color.rgb2hsv(img)
    hsv[:, :, 2] = exposure.equalize_hist(hsv[:, :, 2])
    img = color.hsv2rgb(hsv)

    # central square crop
    min_side = min(img.shape[:-1])
    centre = img.shape[0] // 2, img.shape[1] // 2
    img = img[centre[0] - min_side // 2:centre[0] + min_side // 2,
              centre[1] - min_side // 2:centre[1] + min_side // 2,
              :]

    # rescale to standard size
    img = transform.resize(img, (self.IMG_SIZE, self.IMG_SIZE))

    return img


def cv2_preprocess_img(img, img_size):
    """
    # grayscaling
    gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # contrast limited adaptive histogram equalization
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_img = clahe.apply(gray_img)
    """
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    lab_planes = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab_planes[0] = clahe.apply(lab_planes[0])
    lab = cv2.merge(lab_planes)
    clahe_img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    # normalize image intensities
    img = 2.0*(clahe_img*1.0 / 255) - 1.0
    min_side = min(img.shape[:-1])
    centre = img.shape[0] // 2, img.shape[1] // 2
    img = img[centre[0] - min_side // 2:centre[0] + min_side // 2,
              centre[1] - min_side // 2:centre[1] + min_side // 2]
    img = cv2.resize(img, (img_size, img_size))
    # img = img.reshape(self.IMG_SIZE, self.IMG_SIZE, 1)
    return img


class GtsrbModel:

    """ source dir is the relative path of gtsrb data set' """
    def __init__(self, source_dir=None, start_point=0, epoch=30):
        # Config related to images in the gtsrb dataset
        self.start_point = start_point

        self.num_classes = 43        
        self.IMG_SIZE = 48
        self.epoch = epoch
        self.name = "gtsrb"
        self.batch_size = 256
        self.input_shape = (self.IMG_SIZE, self.IMG_SIZE, 3)
        self.script_path = os.path.dirname(os.path.abspath(__file__))
        self.root_dir = os.path.join(self.script_path, source_dir)
        if not os.path.isdir(self.root_dir):
            logger.fatal("Please download the dataset first.")

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

    def def_model(self, model_id=0):
        input_shape = self.input_shape
        if model_id == 0:
            oxford_model = Sequential()
            oxford_model.add(Conv2D(32, (3, 3), padding="same", activation="relu", input_shape=input_shape))
            oxford_model.add(Conv2D(32, (3, 3), activation="relu"))
            oxford_model.add(MaxPooling2D(pool_size=(2, 2)))
            oxford_model.add(Dropout(0.2))

            oxford_model.add(Conv2D(64, (3, 3), padding="same", activation="relu"))
            oxford_model.add(Conv2D(64, (3, 3), activation="relu"))
            oxford_model.add(MaxPooling2D(pool_size=(2, 2)))
            oxford_model.add(Dropout(0.2))

            oxford_model.add(Conv2D(128, (3, 3), padding="same", activation="relu"))
            oxford_model.add(Conv2D(128, (3, 3), activation="relu"))
            oxford_model.add(MaxPooling2D(pool_size=(2, 2)))
            oxford_model.add(Dropout(0.2))

            if config.enable_filters:
                oxford_model.add(Conv2D(128, (3, 3), padding="same", activation="relu"))
                oxford_model.add(Conv2D(128, (3, 3), activation="relu"))
                oxford_model.add(MaxPooling2D(pool_size=(2, 2)))
                oxford_model.add(Dropout(0.2))

            oxford_model.add(Flatten())
            oxford_model.add(Dense(512, activation='relu'))
            oxford_model.add(Dropout(0.5))
            oxford_model.add(Dense(self.num_classes, activation='softmax'))
            oxford_model.compile(loss=keras.losses.categorical_crossentropy,
                                 optimizer=keras.optimizers.Adam(),
                                 metrics=['accuracy'])
            return_model = oxford_model
        elif model_id == 1:
            vggmodel = applications.VGG16(include_top=False, weights='imagenet', input_shape=input_shape)

            top_model = Sequential()
            top_model.add(Flatten(input_shape=vggmodel.output_shape[1:]))
            top_model.add(Dense(512, activation='relu'))
            top_model.add(Dropout(0.5))
            top_model.add(Dense(256, activation='relu'))
            top_model.add(Dropout(0.5))
            top_model.add(Dense(self.num_classes, activation='softmax'))

            return_model = Model(input=vggmodel.input, output=top_model(vggmodel.output))

            return_model.compile(loss=keras.losses.categorical_crossentropy,
                                 optimizer=keras.optimizers.Adam(lr=1e-5, decay=0.0),
                                 metrics=['accuracy'])
        else:
            raise Exception("unsupported model")
        return return_model

    def load_original_data(self, data_type='train'):
        if data_type == 'train':
            data_dir = self.root_dir+'/Final_Training/train/'
            all_img_path = glob.glob(os.path.join(data_dir, '*/*.ppm'))
        else:
            data_dir = self.root_dir+'/Final_Training/validate/'
            all_img_path = glob.glob(os.path.join(data_dir, '*.ppm'))

        np.random.seed(0)
        np.random.shuffle(all_img_path)        

        imgs = []
        labels = []
        for img_path in all_img_path:
            img = cv2.imread(img_path)
            imgs.append(img)
            if data_type == 'train':
                label = int(img_path.split('/')[-2])
            else:
                filename = os.path.basename(img_path)
                label = int(filename.split('_')[0])
            labels.append(label)
        # Make one hot targets
        labels = np.eye(self.num_classes, dtype='uint8')[labels]
        return imgs, labels

    def load_original_test_data(self):
        data_path = self.root_dir+'/Final_Test/Images/'
        import pandas as pd
        test = pd.read_csv(data_path+'/GT-final_test.csv', sep=';')

        # Load test dataset
        x_test = []
        y_test = []
        for file_name, class_id in zip(list(test['Filename']), list(test['ClassId'])):
            img_path = os.path.join(data_path, file_name)
            x_test.append(cv2.imread(img_path))
            y_test.append(class_id)
        y_test = np.eye(self.num_classes, dtype='uint8')[np.array(y_test)]
        return x_test, y_test

    def preprocess_original_imgs(self, imgs=None):
        for i in range(len(imgs)):
            imgs[i] = cv2_preprocess_img(imgs[i], self.IMG_SIZE)
        # tasks = []
        # for index in range(len(imgs)):   # define tasks
        #     tasks.append((imgs[index], self.IMG_SIZE,))
        # # Run tasks
        # results = [config.pool.apply_async(cv2_preprocess_img, t) for t in tasks]
        # for i in range(len(results)):       # number of test
        #     imgs[i] = results[i].get()
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
        # lr_scheduler = LearningRateScheduler(lr_schedule)
        # lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
        #                                cooldown=0,
        #                                patience=5,
        #                                min_lr=0.5e-6)
        checkpoint = ModelCheckpoint(weights_file, monitor='acc', verbose=1,
                                     save_best_only=False, mode='max')
        callbacks_list = [checkpoint]
        # callbacks_list = [checkpoint, lr_reducer, lr_scheduler]

        x_val = self.preprocess_original_imgs(x_val)

        # if train_strategy is not None:
        #     selector = Selector(model, train_strategy)
        #     callbacks_list.append(selector)
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
            
        if model_id == 0:
            model.fit_generator(data,
                                validation_data=(x_val, y_val),
                                epochs=epochs, verbose=1,
                                callbacks=callbacks_list)
        elif model_id == 1:
            for layer in model.layers:
                layer.trainable = True
            model.load_weights(weights_file)
            model.fit_generator(data,
                                validation_data=(x_val, y_val),
                                epochs=epochs, verbose=1,
                                shuffle=False,
                                callbacks=callbacks_list)

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
    md = GtsrbModel(source_dir='GTSRB')
       
    _models = [(0, "models/gtsrb.oxford.model0.hdf5"), (1, "models/gtsrb.vgg16.model1.hdf5")]
    # 0.9616785431229115
    # 0.9038004751065754
    # 0.971496437026316
    _model0 = _models[0]
    # x_train, y_train = md.load_original_data('train')
    # x_val, y_val = md.load_original_data('val')
    x_original_test, y_original_test = md.load_original_test_data()

    # model=md.train_dnn_model(model_id=_model[0],weights_file=_model[1],
    #                          x_train=md.preprocess_original_imgs(x_train), y_train=y_train,
    #                          x_val=md.preprocess_original_imgs(x_val), y_val=y_val)

    model0 = md.load_model(_model0[0], _model0[1])
    md.test_dnn_model(model0, md.preprocess_original_imgs(x_original_test), y_original_test)
