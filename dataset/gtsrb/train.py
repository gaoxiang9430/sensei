'''
This program is designed to train models for gstrb dataset
Author: Xiang Gao (xiang.gao@us.fujitsu.com)
Time: Sep, 21, 2018
'''

from __future__ import print_function

import os
import glob
import math
import numpy as np
import numpy
import csv
from sklearn.metrics import accuracy_score
import keras
from keras import applications
from keras.datasets import mnist
from keras.models import Model, Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, Convolution2D
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint,LearningRateScheduler
from skimage import color, exposure, transform
from skimage import io
import cv2
import pickle

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

class gtsrb_model():
    'source dir is the relative path of gtsrb data set'
    def __init__(self, source_dir=None):
        # Config related to images in the gstrb dataset
        self.num_classes = 43        
        self.IMG_SIZE = 48

        self.script_path = os.path.dirname(os.path.abspath( __file__ ))
        self.root_dir = os.path.join(self.script_path, source_dir)

        #define a global variable
        self.datagen_rotation = ImageDataGenerator(
            featurewise_center=False, # set input mean to 0 over the dataset
            samplewise_center=False, # set each sample mean to 0
            featurewise_std_normalization=False, # divide inputs by std of dataset
            samplewise_std_normalization=False,# divide each input by its std
            zca_whitening=False,# apply ZCA whitening
            zca_epsilon=1e-06,# epsilon for ZCA whitening
            rotation_range=30,# randomly rotate images in the range (deg 0 to 180)
            width_shift_range=0.1,# randomly shift images horizontally
            height_shift_range=0.1,# randomly shift images vertically
            shear_range=0.2,# set range for random shear
            zoom_range=0.,# set range for random zoom
            channel_shift_range=0.,# set range for random channel shifts
            fill_mode='nearest', # set mode for filling points outside the input boundaries
            cval=0.,# value used for fill_mode = "constant"
            horizontal_flip=True,# randomly flip images
            vertical_flip=False,# randomly flip images
            rescale=None,# set rescaling factor (applied before any other transformation)
            preprocessing_function=None,# set function that will be applied on each input
            data_format=None,# image data format, either "channels_first" or "channels_last"
            validation_split=0.0# fraction of images reserved for validation (strictly between 0 and 1)
        )

    def cv2_preprocess_img(self, img):
        '''
        # grayscaling
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        # contrast limited adaptive histogram equalization
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        clahe_img = clahe.apply(gray_img)
        '''
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        lab_planes = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8, 8))
        lab_planes[0] = clahe.apply(lab_planes[0])
        lab = cv2.merge(lab_planes)
        clahe_img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        # normalize image intensities
        img = 2.0*(clahe_img*1.0 / 255) - 1.0
        min_side = min(img.shape[:-1])
        centre = img.shape[0] // 2, img.shape[1] // 2
        img = img[centre[0] - min_side // 2:centre[0] + min_side // 2,
                  centre[1] - min_side // 2:centre[1] + min_side // 2]
        img = cv2.resize(img, (self.IMG_SIZE, self.IMG_SIZE))
        #img = img.reshape(self.IMG_SIZE, self.IMG_SIZE, 1)
        return img


    def def_model(self, model_id=0):
        input_shape = (self.IMG_SIZE, self.IMG_SIZE, 3)
        if model_id == 0:
            model = Sequential()
            model.add(Conv2D(32, (3, 3), padding="same", activation="relu", input_shape=input_shape))
            model.add(Conv2D(32, (3, 3), activation="relu"))
            model.add(MaxPooling2D(pool_size=(2, 2)))
            model.add(Dropout(0.2))

            model.add(Conv2D(64, (3, 3), padding="same", activation="relu"))
            model.add(Conv2D(64, (3, 3), activation="relu"))
            model.add(MaxPooling2D(pool_size=(2, 2)))
            model.add(Dropout(0.2))

            model.add(Conv2D(128, (3, 3), padding="same", activation="relu"))
            model.add(Conv2D(128, (3, 3), activation="relu"))
            model.add(MaxPooling2D(pool_size=(2, 2)))
            model.add(Dropout(0.2))

            model.add(Flatten()) 
            model.add(Dense(512, activation='relu'))
            model.add(Dropout(0.5))
            model.add(Dense(self.num_classes, activation='softmax'))
            model.compile(loss=keras.losses.categorical_crossentropy,
                      optimizer=keras.optimizers.Adam(),
                      metrics=['accuracy'])
        elif model_id == 1:
            vggmodel = applications.VGG16(include_top=False, weights='imagenet', input_shape=input_shape)

            top_model = Sequential()
            top_model.add(Flatten(input_shape=vggmodel.output_shape[1:]))
            top_model.add(Dense(512, activation='relu'))
            top_model.add(Dropout(0.5))
            top_model.add(Dense(256, activation='relu'))
            top_model.add(Dropout(0.5))
            top_model.add(Dense(self.num_classes, activation='softmax'))

            model = Model(input=vggmodel.input, output=top_model(vggmodel.output))

            model.compile(loss=keras.losses.categorical_crossentropy,
                              optimizer=keras.optimizers.Adam(lr=1e-5, decay=0.0),
                              metrics=['accuracy'])
        return model

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
            #img = self.cv2_preprocess_img(cv2.imread(img_path))
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
            #x_test.append(preprocess_img(io.imread(img_path)))
            x_test.append(cv2.imread(img_path))
            y_test.append(class_id)
        y_test = np.eye(self.num_classes, dtype='uint8')[np.array(y_test)]
        return x_test, y_test

    def preprocess_original_imgs(self, imgs=None):
        for i in range(len(imgs)):
            imgs[i] = self.cv2_preprocess_img(imgs[i])
        imgs = np.array(imgs, dtype='float32')
        return imgs

    def train_dnn_model(self, model_id=0, weights_file="gtsrb.hdf5", x_train=None, y_train=None, x_val=None, y_val=None, data_generator=None):
        '''
        train or test a dnn model on gstrb dataset
        action can have two values: "train", "test"
        '''
        # training config
        epochs = 15
        batch_size = 32

        K.set_image_data_format('channels_last')

        model = self.def_model(model_id)
        
        weights_file = os.path.join(self.script_path, weights_file)
        checkpoint = ModelCheckpoint(weights_file, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
        callbacks_list = [checkpoint]
        if data_generator == None:
            data = self.datagen_rotation.flow(x_train, y_train, batch_size=batch_size)
        else:
            data = data_generator
            
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
                                callbacks=callbacks_list)

        #score = model.evaluate(x_train, y_train, verbose=0)
        #print('Train loss:', score[0])
        #print('Train accuracy:', score[1])
     
        return model

    def load_model(self, model_id=0, weights_file='gtsrb.hdf5'):
        model = self.def_model(model_id)
        weights_file = os.path.join(self.script_path, weights_file)
        model.load_weights(weights_file)
        return model

    def test_dnn_model(self, model=None, x_test=None, y_test=None, print_label=""):
        score = model.evaluate(x_test, y_test, verbose=0)
        print(print_label, ' - Test loss:', score[0])
        print(print_label, ' - Test accuracy:', score[1])


if __name__ == '__main__':
    md = gtsrb_model(source_dir='GTSRB')
       
    _models = [(0, "models/gtsrb.oxford.model0.hdf5"), (1, "models/gtsrb.vgg16.model1.hdf5")]
    # 0.9616785431229115
    # 0.9038004751065754
    # 0.971496437026316
    _model = _models[0]
    #x_train, y_train = md.load_original_data('train')
    #x_val, y_val = md.load_original_data('val')
    x_test, y_test = md.load_original_test_data()

    #model=md.train_dnn_model(model_id=_model[0],weights_file=_model[1],x_train=md.preprocess_original_imgs(x_train),
     #                        y_train=y_train, x_val=md.preprocess_original_imgs(x_val), y_val=y_val)

    model = md.load_model(_model[0], _model[1])
    md.test_dnn_model(model, md.preprocess_original_imgs(x_test), y_test)



