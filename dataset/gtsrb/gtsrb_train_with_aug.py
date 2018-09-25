'''
Reproduce deepknn paper for gtsrb dataset
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
from keras.callbacks import ModelCheckpoint
from skimage import color, exposure, transform
from skimage import io
import cv2
import pickle
from lib.spacial_transformation import *

def hamming2(s1, s2):
    """Calculate the Hamming distance between two bit strings"""
    assert len(s1) == len(s2)
    return sum(c1 != c2 for c1, c2 in zip(s1, s2))

class gtsrb_dknn():
    def __init__(self, index_folder = "./"):
        self.index_folder = index_folder
        self.index = []
        self.a_x_y = []

        self.num_classes = 43        
        self.IMG_SIZE = 48

        root_dir = 'GTSRB/Final_Training/Images/'
        imgs = []
        labels = []
        self.x_train = []
        self.x_test = []
        
    def readimage(self, img_path):
        return io.imread(img_path)

    def preprocess(self, cv2read_images):
        '''
        This function is called from outside
        It is intended to be called by ga_realistic_transform.py
        '''
        x_test = []
        for img in cv2read_images:
            # convert BGR to RGB and then preprocess the image
            img1 = np.array(img[:,:,::-1], dtype='uint8')
            x_test.append(self.preprocess_img(img1))
        x_test = np.array(x_test, dtype='float32')
        return x_test

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

        # roll color axis to axis 0
        #img = np.rollaxis(img, -1)

        return img


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


    def get_class(self, img_path):
        return int(img_path.split('/')[-2])

    def load_dnn_model(self):
        '''
        load a pretrained dnn model and assign the model to self.model
        '''
        num_classes = 43
        IMG_SIZE = 48
        input_shape = (IMG_SIZE, IMG_SIZE, 3)
        model = Sequential()
        model.add(Conv2D(64, kernel_size=(8, 8), strides=2, padding='same',
                         activation='relu',
                         input_shape=input_shape))
        model.add(Conv2D(128, kernel_size=(6, 6), strides=2, padding='valid',
                         activation='relu'))
        model.add(Conv2D(128, (5, 5), strides=1, padding='valid', activation='relu'))
        #model.add(MaxPooling2D(pool_size=(2, 2)))
        #model.add(Dropout(0.25))
        model.add(Flatten())
        #model.add(Dense(128, activation='relu'))
        #model.add(Dropout(0.5))
        model.add(Dense(200, activation='relu'))
        model.add(Dense(num_classes, activation='softmax'))

        model.compile(loss=keras.losses.categorical_crossentropy,
                      optimizer=keras.optimizers.Adam(),
                      metrics=['accuracy'])
        model.load_weights("gtsrb.weights.best.hdf5")
        self.model = model

    def train_dnn(self, action = "test"):
        '''
        train or test a dnn model on mnist dataset
        action can have two values: "train", "test"
        '''
        num_classes = 43
        batch_size = 32
        epochs = 30
        IMG_SIZE = 48
        input_shape = (IMG_SIZE, IMG_SIZE, 3)
        root_dir = 'GTSRB/Final_Training/Images/'
        imgs = []
        labels = []
        if action == "train":
            all_img_paths = glob.glob(os.path.join(root_dir, '*/*.ppm'))
            np.random.shuffle(all_img_paths)
            for img_path in all_img_paths:
                #img = self.preprocess_img(io.imread(img_path))
                img = self.cv2_preprocess_img(cv2.imread(img_path))
                label = self.get_class(img_path)
                imgs.append(img)
                labels.append(label)

            x_train = np.array(imgs, dtype='float32')
            # Make one hot targets
            y_train = np.eye(num_classes, dtype='uint8')[labels]


        import pandas as pd
        test = pd.read_csv('GTSRB/GT-final_test.csv', sep=';')

        # Load test dataset
        x_test = []
        y_test = []
        i = 0
        for file_name, class_id in zip(list(test['Filename']), list(test['ClassId'])):
            img_path = os.path.join('GTSRB/Final_Test/Images/', file_name)
            #x_test.append(self.preprocess_img(io.imread(img_path)))
            x_test.append(self.preprocess_img(cv2.imread(img_path)[:,:,::-1]))
            y_test.append(class_id)

        x_test = np.array(x_test, dtype='float32')
        y_test = np.eye(num_classes, dtype='uint8')[np.array(y_test)]

        K.set_image_data_format('channels_last')

        model = Sequential()
        model.add(Conv2D(64, kernel_size=(8, 8), strides=2, padding='same',
                         activation='relu',
                         input_shape=input_shape))
        model.add(Conv2D(128, kernel_size=(6, 6), strides=2, padding='valid',
                         activation='relu'))
        model.add(Conv2D(128, (5, 5), strides=1, padding='valid', activation='relu'))
        #model.add(MaxPooling2D(pool_size=(2, 2)))
        #model.add(Dropout(0.25))
        model.add(Flatten())
        #model.add(Dense(128, activation='relu'))
        #model.add(Dropout(0.5))
        model.add(Dense(200, activation='relu'))
        model.add(Dense(num_classes, activation='softmax'))

        model.compile(loss=keras.losses.categorical_crossentropy,
                      optimizer=keras.optimizers.Adam(),
                      metrics=['accuracy'])
        if action == "train":
            filepath="gtsrb.weights.best.hdf5"
            checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
            callbacks_list = [checkpoint]

            model.fit(x_train, y_train,
                      batch_size=batch_size,
                      epochs=epochs, 
                      callbacks=callbacks_list,
                      verbose=1,
                      validation_data=(x_test, y_test))

        #x_test = np.array([x_test[0]])
        #y_test = np.array([y_test[0]])

        score = model.evaluate(x_test, y_test, verbose=0)
        print('Test loss:', score[0])
        print('Test accuracy:', score[1])
        model.load_weights("gtsrb.weights.best.hdf5")
        score = model.evaluate(x_test, y_test, verbose=0)
        print('Test loss:', score[0])
        print('Test accuracy:', score[1])

    def train_dnn_model(self, action = "test", modelid=0, weights_file="gtsrb.hdf5"):
        '''
        train or test a dnn model on mnist dataset
        action can have two values: "train", "test"
        '''
        num_classes = 43
        batch_size = 32
        epochs = 30
        IMG_SIZE = 48
        input_shape = (IMG_SIZE, IMG_SIZE, 3)
        root_dir = 'GTSRB/Final_Training/train/'
        val_dir = 'GTSRB/Final_Training/validate/'
        imgs = []
        labels = []
        if action == "train":
            all_img_paths = glob.glob(os.path.join(root_dir, '*/*.ppm'))
            #print("number of training images")
            #print(len(all_img_paths))

            #exit()
            
            np.random.seed(0)
            np.random.shuffle(all_img_paths)
            for img_path in all_img_paths:
                #img = self.preprocess_img(io.imread(img_path))
                img = self.cv2_preprocess_img(cv2.imread(img_path))
                label = self.get_class(img_path)
                imgs.append(img)
                labels.append(label)

            x_train = np.array(imgs, dtype='float32')
            # Make one hot targets
            y_train = np.eye(num_classes, dtype='uint8')[labels]

            print(x_train.shape)

            # Load val dataset
            x_val = []
            y_val = []
            val_imgs = []
            val_labels = []
            val_img_paths = glob.glob(os.path.join(val_dir, '*.ppm'))

            i = 0
            for val_img_path in val_img_paths:

                val_imgs.append(self.cv2_preprocess_img(cv2.imread(val_img_path)))
                filename = os.path.basename(val_img_path)
                val_labels.append(int(filename.split('_')[0]))

            x_val = np.array(val_imgs, dtype='float32')
            y_val = np.eye(num_classes, dtype='uint8')[val_labels]
            print(x_val.shape)

        import pandas as pd
        test = pd.read_csv('GTSRB/GT-final_test.csv', sep=';')

        # Load test dataset
        x_test = []
        y_test = []
        i = 0
        for file_name, class_id in zip(list(test['Filename']), list(test['ClassId'])):
            img_path = os.path.join('GTSRB/Final_Test/Images/', file_name)
            #x_test.append(self.preprocess_img(io.imread(img_path)))
            x_test.append(self.cv2_preprocess_img(cv2.imread(img_path)))
            y_test.append(class_id)

        x_test = np.array(x_test, dtype='float32')
        y_test = np.eye(num_classes, dtype='uint8')[np.array(y_test)]

        K.set_image_data_format('channels_last')

        
        if modelid == 0:
            epochs = 30
            model = Sequential()

            model.add(Convolution2D(32, 3, 3, border_mode='same', input_shape=input_shape, activation='relu'))
            model.add(Convolution2D(32, 3, 3, activation='relu'))
            model.add(MaxPooling2D(pool_size=(2, 2)))
            model.add(Dropout(0.2))

            model.add(Convolution2D(64, 3, 3, border_mode='same', activation='relu'))
            model.add(Convolution2D(64, 3, 3, activation='relu'))
            model.add(MaxPooling2D(pool_size=(2, 2)))
            model.add(Dropout(0.2))

            model.add(Convolution2D(128, 3, 3, border_mode='same', activation='relu'))
            model.add(Convolution2D(128, 3, 3, activation='relu'))
            model.add(MaxPooling2D(pool_size=(2, 2)))
            model.add(Dropout(0.2))

            model.add(Flatten())
            model.add(Dense(512, activation='relu'))
            model.add(Dropout(0.5))
            model.add(Dense(num_classes, activation='softmax'))
            model.compile(loss=keras.losses.categorical_crossentropy,
                      optimizer=keras.optimizers.Adam(),
                      metrics=['accuracy'])
        elif modelid == 1:
            epochs = 30
            vggmodel = applications.VGG16(include_top=False, weights='imagenet', input_shape=input_shape)
            
            top_model = Sequential()
            top_model.add(Flatten(input_shape=vggmodel.output_shape[1:]))
            top_model.add(Dense(512, activation='relu'))
            top_model.add(Dropout(0.5))
            top_model.add(Dense(256, activation='relu'))
            top_model.add(Dropout(0.5))
            top_model.add(Dense(num_classes, activation='softmax'))

            model = Model(input=vggmodel.input, output=top_model(vggmodel.output))
            for layer in model.layers[:13]:
                layer.trainable = False

            #print(model.summary())
            model.compile(loss=keras.losses.categorical_crossentropy,
                          optimizer=keras.optimizers.Adam(lr=1e-5, decay=0.0),
                          metrics=['accuracy'])

        elif modelid == 2:
            epochs = 30
            model = Sequential()

            model.add(Convolution2D(100, 3, 3, border_mode='valid', input_shape=input_shape, activation='relu'))
            model.add(MaxPooling2D(pool_size=(2, 2)))            
            model.add(Convolution2D(150, 4, 4, border_mode='valid', activation='relu'))
            model.add(MaxPooling2D(pool_size=(2, 2)))

            model.add(Convolution2D(250, 3, 3, border_mode='valid', activation='relu'))
            model.add(MaxPooling2D(pool_size=(2, 2)))

            model.add(Flatten())
            model.add(Dense(200, activation='relu'))
            model.add(Dense(num_classes, activation='softmax'))
            model.compile(loss=keras.losses.categorical_crossentropy,
                      optimizer=keras.optimizers.Adam(),
                      metrics=['accuracy'])
        
        if action == "train":
            filepath = weights_file
            checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
            callbacks_list = [checkpoint]

            model.fit(x_train, y_train,
                      batch_size=batch_size,
                      epochs=epochs, 
                      callbacks=callbacks_list,
                      verbose=1,
                      validation_data=(x_val, y_val))

        #x_test = np.array([x_test[0]])
        #y_test = np.array([y_test[0]])

        score = model.evaluate(x_test, y_test, verbose=0)
        print('Test loss:', score[0])
        print('Test accuracy:', score[1])
        model.load_weights(weights_file)
        score = model.evaluate(x_test, y_test, verbose=0)
        print('Test loss:', score[0])
        print('Test accuracy:', score[1])


    def test_dnn_model_predict(self, modelid=0, weights_file="gtsrb.hdf5", trans=None):
        '''
        train or test a dnn model on mnist dataset
        action can have two values: "train", "test"
        '''

        num_classes = 43
        batch_size = 32
        epochs = 30
        IMG_SIZE = 48
        input_shape = (IMG_SIZE, IMG_SIZE, 3)
        root_dir = 'GTSRB/Final_Training/train/'
        data_output = []
        all_accuracy = []
        all_preds = []

        all_img_paths = glob.glob(os.path.join(root_dir, '*/*.ppm'))            
        np.random.seed(0)
        np.random.shuffle(all_img_paths)

        for degree in xrange(trans[1]):
            imgs = []
            labels = []
            data_output_degree = []
            print(trans[0] + ' ' + str(trans[3]+degree*trans[2]))

            for img_path in all_img_paths:
                #img = self.preprocess_img(io.imread(img_path))
                rotated_img_path = img_path.replace("train", "Images" + "_" + trans[0] + "_" + str(trans[3]+degree*trans[2]))
                img = self.cv2_preprocess_img(cv2.imread(rotated_img_path))
                label = self.get_class(rotated_img_path)
                imgs.append(img)
                labels.append(label)

            x_train = np.array(imgs, dtype='float32')
            # Make one hot targets
            y_train = np.eye(num_classes, dtype='uint8')[labels]
            y_train_1 = labels


            

            K.set_image_data_format('channels_last')

            if modelid == 0:
                epochs = 30
                model = Sequential()

                model.add(Convolution2D(32, 3, 3, border_mode='same', input_shape=input_shape, activation='relu'))
                model.add(Convolution2D(32, 3, 3, activation='relu'))
                model.add(MaxPooling2D(pool_size=(2, 2)))
                model.add(Dropout(0.2))

                model.add(Convolution2D(64, 3, 3, border_mode='same', activation='relu'))
                model.add(Convolution2D(64, 3, 3, activation='relu'))
                model.add(MaxPooling2D(pool_size=(2, 2)))
                model.add(Dropout(0.2))

                model.add(Convolution2D(128, 3, 3, border_mode='same', activation='relu'))
                model.add(Convolution2D(128, 3, 3, activation='relu'))
                model.add(MaxPooling2D(pool_size=(2, 2)))
                model.add(Dropout(0.2))

                model.add(Flatten())
                model.add(Dense(512, activation='relu'))
                model.add(Dropout(0.5))
                model.add(Dense(num_classes, activation='softmax'))
                model.compile(loss=keras.losses.categorical_crossentropy,
                          optimizer=keras.optimizers.Adam(),
                          metrics=['accuracy'])
            elif modelid == 1:
                epochs = 30
                vggmodel = applications.VGG16(include_top=False, weights='imagenet', input_shape=input_shape)
                
                top_model = Sequential()
                top_model.add(Flatten(input_shape=vggmodel.output_shape[1:]))
                top_model.add(Dense(512, activation='relu'))
                top_model.add(Dropout(0.5))
                top_model.add(Dense(256, activation='relu'))
                top_model.add(Dropout(0.5))
                top_model.add(Dense(num_classes, activation='softmax'))

                model = Model(input=vggmodel.input, output=top_model(vggmodel.output))
                for layer in model.layers[:13]:
                    layer.trainable = False

                #print(model.summary())
                model.compile(loss=keras.losses.categorical_crossentropy,
                              optimizer=keras.optimizers.Adam(lr=1e-5, decay=0.0),
                              metrics=['accuracy'])

            elif modelid == 2:
                epochs = 30
                model = Sequential()

                model.add(Convolution2D(100, 3, 3, border_mode='valid', input_shape=input_shape, activation='relu'))
                model.add(MaxPooling2D(pool_size=(2, 2)))            
                model.add(Convolution2D(150, 4, 4, border_mode='valid', activation='relu'))
                model.add(MaxPooling2D(pool_size=(2, 2)))

                model.add(Convolution2D(250, 3, 3, border_mode='valid', activation='relu'))
                model.add(MaxPooling2D(pool_size=(2, 2)))

                model.add(Flatten())
                model.add(Dense(200, activation='relu'))
                model.add(Dense(num_classes, activation='softmax'))
                model.compile(loss=keras.losses.categorical_crossentropy,
                          optimizer=keras.optimizers.Adam(),
                          metrics=['accuracy'])
        
           

            model.load_weights(weights_file)
            accuracy = 0
            preds = 0

            preds = model.predict(x_train)
            y_preds = np.argmax(preds, axis = 1)
            output = np.equal(y_preds, y_train_1)*1
            accuracy = np.mean(output)
            data_output.append(output)
            all_preds.append(y_preds)
            all_accuracy.append(accuracy)         
        return data_output, all_accuracy, all_preds
    
    def train_dnn_model_aug3(self, action = "test", modelid=0, weights_file="gtsrb.hdf5"):
        '''
        train or test a dnn model on mnist dataset
        action can have two values: "train", "test"
        '''
        num_classes = 43
        batch_size = 32
        epochs = 30
        IMG_SIZE = 48
        input_shape = (IMG_SIZE, IMG_SIZE, 3)
        root_dir = 'GTSRB/Final_Training/train/'
        val_dir = 'GTSRB/Final_Training/validate/'
        imgs = []
        labels = []
        if action == "train":
            all_img_paths = glob.glob(os.path.join(root_dir, '*/*.ppm'))
            #print("number of training images")
            #print(len(all_img_paths))

            #exit()
            
            np.random.seed(0)
            np.random.shuffle(all_img_paths)
            for img_path in all_img_paths:
                #img = self.preprocess_img(io.imread(img_path))
                img = self.cv2_preprocess_img(cv2.imread(img_path))
                label = self.get_class(img_path)
                imgs.append(img)
                labels.append(label)

            x_train = np.array(imgs, dtype='float32')
            # Make one hot targets
            y_train = np.eye(num_classes, dtype='uint8')[labels]

            print(x_train.shape)

            # Load val dataset
            x_val = []
            y_val = []
            val_imgs = []
            val_labels = []
            val_img_paths = glob.glob(os.path.join(val_dir, '*.ppm'))

            i = 0
            for val_img_path in val_img_paths:

                val_imgs.append(self.cv2_preprocess_img(cv2.imread(val_img_path)))
                filename = os.path.basename(val_img_path)
                val_labels.append(int(filename.split('_')[0]))

            x_val = np.array(val_imgs, dtype='float32')
            y_val = np.eye(num_classes, dtype='uint8')[val_labels]
            print(x_val.shape)

        import pandas as pd
        test = pd.read_csv('GTSRB/GT-final_test.csv', sep=';')

        # Load test dataset
        x_test = []
        y_test = []
        i = 0
        for file_name, class_id in zip(list(test['Filename']), list(test['ClassId'])):
            img_path = os.path.join('GTSRB/Final_Test/Images/', file_name)
            #x_test.append(self.preprocess_img(io.imread(img_path)))
            x_test.append(self.cv2_preprocess_img(cv2.imread(img_path)))
            y_test.append(class_id)

        x_test = np.array(x_test, dtype='float32')
        y_test = np.eye(num_classes, dtype='uint8')[np.array(y_test)]

        K.set_image_data_format('channels_last')

        
        if modelid == 0:
            epochs = 30
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
            model.add(Dense(num_classes, activation='softmax'))
            model.compile(loss=keras.losses.categorical_crossentropy,
                      optimizer=keras.optimizers.Adam(),
                      metrics=['accuracy'])
        elif modelid == 1:
            epochs = 30
            vggmodel = applications.VGG16(include_top=False, weights='imagenet', input_shape=input_shape)
            
            top_model = Sequential()
            top_model.add(Flatten(input_shape=vggmodel.output_shape[1:]))
            top_model.add(Dense(512, activation='relu'))
            top_model.add(Dropout(0.5))
            top_model.add(Dense(256, activation='relu'))
            top_model.add(Dropout(0.5))
            top_model.add(Dense(num_classes, activation='softmax'))

            model = Model(input=vggmodel.input, output=top_model(vggmodel.output))

            model.compile(loss=keras.losses.categorical_crossentropy,
                              optimizer=keras.optimizers.Adam(lr=1e-5, decay=0.0),
                              metrics=['accuracy'])

        
        if action == "train":
            filepath = weights_file

            datagen_rotation = ImageDataGenerator(
            # set input mean to 0 over the dataset
            featurewise_center=False,
            # set each sample mean to 0
            samplewise_center=False,
            # divide inputs by std of dataset
            featurewise_std_normalization=False,
            # divide each input by its std
            samplewise_std_normalization=False,
            # apply ZCA whitening
            zca_whitening=False,
            # epsilon for ZCA whitening
            zca_epsilon=1e-06,
            # randomly rotate images in the range (deg 0 to 180)
            rotation_range=30,
            # randomly shift images horizontally
            width_shift_range=0.1,
            # randomly shift images vertically
            height_shift_range=0.1,
            # set range for random shear
            shear_range=0.2,
            # set range for random zoom
            zoom_range=0.,
            # set range for random channel shifts
            channel_shift_range=0.,
            # set mode for filling points outside the input boundaries
            fill_mode='nearest',
            # value used for fill_mode = "constant"
            cval=0.,
            # randomly flip images
            horizontal_flip=True,
            # randomly flip images
            vertical_flip=False,
            # set rescaling factor (applied before any other transformation)
            rescale=None,
            # set function that will be applied on each input
            preprocessing_function=None,
            # image data format, either "channels_first" or "channels_last"
            data_format=None,
            # fraction of images reserved for validation (strictly between 0 and 1)
            validation_split=0.0)
            checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
            callbacks_list = [checkpoint]
            data = datagen_rotation.flow(x_train, y_train, batch_size=batch_size)
            
            if modelid == 0:
                model.fit_generator(data,
                                    validation_data=(x_val, y_val),
                                    epochs=epochs, verbose=1,
                                    callbacks=callbacks_list)
            if modelid == 1:
                '''
                for layer in model.layers[:13]:
                    layer.trainable = False
                
                model.compile(loss=keras.losses.categorical_crossentropy,
                          optimizer=keras.optimizers.Adam(lr=1e-3, decay=0.0),
                          metrics=['accuracy'])

                model.fit_generator(data,
                                    validation_data=(x_val, y_val),
                                    epochs=30, verbose=1,
                                    callbacks=callbacks_list)
                '''
                for layer in model.layers:
                    layer.trainable = True

                #print(model.summary())
                model.compile(loss=keras.losses.categorical_crossentropy,
                              optimizer=keras.optimizers.Adam(lr=1e-5, decay=0.0),
                              metrics=['accuracy'])
                model.load_weights(weights_file)
                model.fit_generator(data,
                                    validation_data=(x_val, y_val),
                                    epochs=30, verbose=1,
                                    callbacks=callbacks_list)


        #x_test = np.array([x_test[0]])
        #y_test = np.array([y_test[0]])

        score = model.evaluate(x_test, y_test, verbose=0)
        print('Test loss:', score[0])
        print('Test accuracy:', score[1])
        model.load_weights(weights_file)
        score = model.evaluate(x_test, y_test, verbose=0)
        print('Test loss:', score[0])
        print('Test accuracy:', score[1])

    def robustness_well(self, modelid=0, weights_file="gtsrb.hdf5", trans=None):
        '''
        train or test a dnn model on mnist dataset
        action can have two values: "train", "test"
        '''
        num_classes = 43
        batch_size = 32
        epochs = 30
        IMG_SIZE = 48
        input_shape = (IMG_SIZE, IMG_SIZE, 3)
        root_dir = 'GTSRB/Final_Training/train/'
        val_dir = 'GTSRB/Final_Training/validate/'
        

        rotation_range = [-30, -25, -20, -15, -10, -5, 0, 5, 10, 15, 20, 25, 30]
        translation_range = [-3, -2, -1, 0, 1, 2, 3]
        shear_range = [-0.2, -0.16, -0.12, -0.08, -0.04, 0, 0.04, 0.08, 0.12, 0.16, 0.2]

        trans_functions = {}
        trans_functions["rotate_c"] = image_rotation_cropped
        trans_functions["translate_c"] = image_translation_cropped
        trans_functions["shear_c"] = image_shear_cropped
        trans_functions["zoom_c"] = image_zoom
        trans_functions["blur_c"] = image_blur
        trans_functions["brightness_c"] = image_brightness
        trans_functions["contrast_c"] = image_contrast

        all_img_paths = glob.glob(os.path.join(root_dir, '*/*.ppm'))
        #print("number of training images")
        #print(len(all_img_paths))
        np.random.seed(0)
        np.random.shuffle(all_img_paths)

        if modelid == 0:
            epochs = 30
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
            model.add(Dense(num_classes, activation='softmax'))
            model.compile(loss=keras.losses.categorical_crossentropy,
                      optimizer=keras.optimizers.Adam(),
                      metrics=['accuracy'])
        elif modelid == 1:
            epochs = 30
            vggmodel = applications.VGG16(include_top=False, weights='imagenet', input_shape=input_shape)
            
            top_model = Sequential()
            top_model.add(Flatten(input_shape=vggmodel.output_shape[1:]))
            top_model.add(Dense(512, activation='relu'))
            top_model.add(Dropout(0.5))
            top_model.add(Dense(256, activation='relu'))
            top_model.add(Dropout(0.5))
            top_model.add(Dense(num_classes, activation='softmax'))

            model = Model(input=vggmodel.input, output=top_model(vggmodel.output))
        model.load_weights(weights_file)
        #exit()
        with open('gtsrb_exp14_3_grid'+str(modelid)+'.csv', 'w',0) as csvfile:
            writer = csv.writer(csvfile, delimiter=',',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
            writer.writerow(['parameter', 'accuracy', 'output_correctness','preds'])        

            for p1 in rotation_range:
                for p2 in translation_range:
                    for p3 in shear_range:
                        print(p1)
                        print(p2)
                        print(p3)
                        
                        imgs = []
                        labels = []
                        
                        for img_path in all_img_paths:
                            #img = self.preprocess_img(io.imread(img_path))
                            img = cv2.imread(img_path)
                            img = trans_functions['rotate_c'](img, p1)
                            img = trans_functions["translate_c"](img, p2)
                            img = trans_functions["shear_c"](img, p3)
                            img = self.cv2_preprocess_img(img)
                            label = self.get_class(img_path)
                            imgs.append(img)
                            labels.append(label)

                        x_train = np.array(imgs, dtype='float32')
                        # Make one hot targets
                        #y_train = np.eye(num_classes, dtype='uint8')[labels]

                        del imgs

                        preds = model.predict(x_train)
                        y_preds = np.argmax(preds, axis = 1)
                        print(y_preds)
                        #print(np.squeeze(self.y_train))
                        output = np.equal(y_preds, np.array(labels))*1
                        accuracy = np.mean(output)
                        print("accuracy" + str(accuracy))
                        #data_output.append(output)
                        #all_preds.append(y_preds)
                        #all_accuracy.append(accuracy)         
                        

                        
                        csvcontent = []
                        csvcontent.append("combined transformation")
                        csvcontent.append(accuracy)
                        output_string = ';'.join(str(x) for x in output)
                        
                        csvcontent.append(output_string)
                        output_string = ';'.join(str(x) for x in y_preds)
                        csvcontent.append(output_string)
                        writer.writerow(csvcontent)

    def adversarial(self, modelid=0, weights_file="gtsrb.hdf5", trans=None):
        '''
        train or test a dnn model on mnist dataset
        action can have two values: "train", "test"
        '''
        num_classes = 43
        batch_size = 32
        epochs = 30
        IMG_SIZE = 48
        input_shape = (IMG_SIZE, IMG_SIZE, 3)
        
        import pandas as pd
        test = pd.read_csv('GTSRB/GT-final_test.csv', sep=';')

        # Load test dataset
        all_img_paths = []
        x_test = []
        y_test = []

        for file_name, class_id in zip(list(test['Filename']), list(test['ClassId'])):
            img_path = os.path.join('GTSRB/Final_Test/Images/', file_name)
            #x_test.append(self.preprocess_img(io.imread(img_path)))
            all_img_paths.append(img_path)
            y_test.append(class_id)

        K.set_image_data_format('channels_last')
        

        rotation_range = [-30, -25, -20, -15, -10, -5, 0, 5, 10, 15, 20, 25, 30]
        translation_range = [-3, -2, -1, 0, 1, 2, 3]
        shear_range = [-0.2, -0.16, -0.12, -0.08, -0.04, 0, 0.04, 0.08, 0.12, 0.16, 0.2]

        trans_functions = {}
        trans_functions["rotate_c"] = image_rotation_cropped
        trans_functions["translate_c"] = image_translation_cropped
        trans_functions["shear_c"] = image_shear_cropped
        trans_functions["zoom_c"] = image_zoom
        trans_functions["blur_c"] = image_blur
        trans_functions["brightness_c"] = image_brightness
        trans_functions["contrast_c"] = image_contrast

        if modelid == 0:
            epochs = 30
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
            model.add(Dense(num_classes, activation='softmax'))
            model.compile(loss=keras.losses.categorical_crossentropy,
                      optimizer=keras.optimizers.Adam(),
                      metrics=['accuracy'])
        elif modelid == 1:
            epochs = 30
            vggmodel = applications.VGG16(include_top=False, weights='imagenet', input_shape=input_shape)
            
            top_model = Sequential()
            top_model.add(Flatten(input_shape=vggmodel.output_shape[1:]))
            top_model.add(Dense(512, activation='relu'))
            top_model.add(Dropout(0.5))
            top_model.add(Dense(256, activation='relu'))
            top_model.add(Dropout(0.5))
            top_model.add(Dense(num_classes, activation='softmax'))

            model = Model(input=vggmodel.input, output=top_model(vggmodel.output))
        model.load_weights(weights_file)

        with open('gtsrb_exp14_test_3_grid'+str(modelid)+'.csv', 'w',0) as csvfile:
            writer = csv.writer(csvfile, delimiter=',',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
            writer.writerow(['parameter', 'accuracy', 'output_correctness','preds'])        

            for p1 in rotation_range:
                for p2 in translation_range:
                    for p3 in shear_range:
                        print(p1)
                        print(p2)
                        print(p3)
                        
                        imgs = []
                        
                        for img_path in all_img_paths:
                            #img = self.preprocess_img(io.imread(img_path))
                            img = cv2.imread(img_path)
                            img = trans_functions['rotate_c'](img, p1)
                            img = trans_functions["translate_c"](img, p2)
                            img = trans_functions["shear_c"](img, p3)
                            img = self.cv2_preprocess_img(img)                            
                            imgs.append(img)
                            
                        
                        x_test = np.array(imgs, dtype='float32')
                        y_test = np.array(y_test)
                        del imgs

                        preds = model.predict(x_test)
                        y_preds = np.argmax(preds, axis = 1)
                        print(y_preds)
                        #print(np.squeeze(self.y_train))
                        output = np.equal(y_preds, y_test)*1
                        accuracy = np.mean(output)
                        print("accuracy" + str(accuracy))
                        
                        csvcontent = []
                        csvcontent.append("combined transformation")
                        csvcontent.append(accuracy)
                        output_string = ';'.join(str(x) for x in output)
                        
                        csvcontent.append(output_string)
                        output_string = ';'.join(str(x) for x in y_preds)
                        csvcontent.append(output_string)
                        writer.writerow(csvcontent)
    
    def getfeaturespace(self, modelid=0, weights_file="gtsrb.hdf5"):
        
        num_classes = 43
        batch_size = 32
        epochs = 30
        IMG_SIZE = 48
        input_shape = (IMG_SIZE, IMG_SIZE, 3)
        root_dir = 'GTSRB/Final_Training/train/'
        val_dir = 'GTSRB/Final_Training/validate/'
        imgs = []
        labels = []
        
        all_img_paths = glob.glob(os.path.join(root_dir, '*/*.ppm'))
        #print("number of training images")
        #print(len(all_img_paths))

        #exit()
        
        np.random.seed(0)
        np.random.shuffle(all_img_paths)
        for img_path in all_img_paths:
            #img = self.preprocess_img(io.imread(img_path))
            img = self.cv2_preprocess_img(cv2.imread(img_path))
            label = self.get_class(img_path)
            imgs.append(img)
            labels.append(label)

        x_train = np.array(imgs, dtype='float32')
        # Make one hot targets
        #y_train = np.eye(num_classes, dtype='uint8')[labels]
        y_train = np.array(labels)
        print(x_train.shape)

        # Load val dataset
        x_val = []
        y_val = []
        val_imgs = []
        val_labels = []
        val_img_paths = glob.glob(os.path.join(val_dir, '*.ppm'))

        i = 0
        for val_img_path in val_img_paths:

            val_imgs.append(self.cv2_preprocess_img(cv2.imread(val_img_path)))
            filename = os.path.basename(val_img_path)
            val_labels.append(int(filename.split('_')[0]))

        x_val = np.array(val_imgs, dtype='float32')
        y_val = np.eye(num_classes, dtype='uint8')[val_labels]
        print(x_val.shape)

        import pandas as pd
        test = pd.read_csv('GTSRB/GT-final_test.csv', sep=';')

        # Load test dataset
        x_test = []
        y_test = []
        i = 0
        for file_name, class_id in zip(list(test['Filename']), list(test['ClassId'])):
            img_path = os.path.join('GTSRB/Final_Test/Images/', file_name)
            #x_test.append(self.preprocess_img(io.imread(img_path)))
            x_test.append(self.cv2_preprocess_img(cv2.imread(img_path)))
            y_test.append(class_id)

        x_test = np.array(x_test, dtype='float32')
        #y_test = np.eye(num_classes, dtype='uint8')[np.array(y_test)]
        y_test = np.array(y_test)
        K.set_image_data_format('channels_last')

        
        if modelid == 0:
            epochs = 30
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
            model.add(Dense(num_classes, activation='softmax'))
            model.compile(loss=keras.losses.categorical_crossentropy,
                      optimizer=keras.optimizers.Adam(),
                      metrics=['accuracy'])
        elif modelid == 1:
            epochs = 30
            vggmodel = applications.VGG16(include_top=False, weights='imagenet', input_shape=input_shape)
            
            top_model = Sequential()
            top_model.add(Flatten(input_shape=vggmodel.output_shape[1:]))
            top_model.add(Dense(512, activation='relu'))
            top_model.add(Dropout(0.5))
            top_model.add(Dense(256, activation='relu'))
            top_model.add(Dropout(0.5))
            top_model.add(Dense(num_classes, activation='softmax'))

            model = Model(input=vggmodel.input, output=top_model(vggmodel.output))

            model.compile(loss=keras.losses.categorical_crossentropy,
                          optimizer=keras.optimizers.Adam(lr=1e-3, decay=0.0),
                          metrics=['accuracy'])

        model.load_weights(weights_file)
        #print(model.summary())
        layer_model = Model(inputs = model.inputs,
                        outputs = model.layers[-2].output)
        x_output = layer_model.predict(x_train)
        save_object(x_output, "gtsrb_layer_x_train_"+str(modelid)+".pkl")
        x_output = layer_model.predict(x_test)
        save_object(x_output, "gtsrb_layer_x_test_"+str(modelid)+".pkl")


        save_object(y_train, "gtsrb_layer_y_train_"+str(modelid)+".pkl")

        save_object(y_test, "gtsrb_layer_y_test_"+str(modelid)+".pkl")
        #print(x_output.shape)
        #model.predict(X_test)

if __name__ == '__main__':
    md = gtsrb_dknn(index_folder = "/data/dataset/dknn/index/")
       
    #model = [(0, "gtsrb.IDSIA.model2.hdf5"), (1, "gtsrb.vgg16.model1.hdf5"),(0, "gtsrb.oxford.model0.hdf5")]
    model = [(0, "gtsrb.oxford.model0.hdf5"), (1, "gtsrb.vgg16.model1.hdf5")]
    # 0.9616785431229115
    # 0.9038004751065754
    # 0.971496437026316
    for m in model:
        if m[0] == 0:
            continue
        #md.train_dnn_model_aug3(action="train", modelid=m[0], weights_file=m[1])

    #for m in model:    
       #md.train_dnn_model_aug3(action="test", modelid=m[0], weights_file=m[1])

    for m in model:
        #md.robustness_well(modelid=m[0], weights_file=m[1])
        #md.adversarial(modelid=m[0], weights_file=m[1])
        md.getfeaturespace(modelid=m[0], weights_file=m[1])


        

    
