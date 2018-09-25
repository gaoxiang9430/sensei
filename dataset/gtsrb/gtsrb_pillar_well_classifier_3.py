'''
Cifar-10 robustness experiment
'''

from __future__ import print_function

import os
import glob
import numpy as np
import csv
import math
import keras
from keras import applications
from keras.models import Model, Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, Convolution2D
from keras import backend as K
from keras.optimizers import SGD, Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, ReduceLROnPlateau
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from keras.wrappers.scikit_learn import KerasClassifier
import cv2
import pickle
import random


def save_object(obj, filename):
    with open(filename, 'wb') as output:
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)

class robust_pillar():


    def __init__(self):
        # 0: well; 1: pillar
        with open("gtsrb_layer_x_train_0.pkl", 'rb') as handle:
            layer_x_train = pickle.load(handle)
        with open("gtsrb_layer_x_test_0.pkl", 'rb') as handle:
            layer_x_test = pickle.load(handle)
        with open("gtsrb_layer_y_train_0.pkl", 'rb') as handle:
            y_train = pickle.load(handle)
        with open("gtsrb_layer_y_test_0.pkl", 'rb') as handle:
            y_test = pickle.load(handle)

        with open('gtsrb_exp14_3_grid0.csv', 'rb') as csvfile:
            myreader = csv.reader(csvfile, delimiter=',', quotechar='|')
            gtsrb_outputs = list(myreader)
        with open('gtsrb_exp14_test_3_grid0.csv', 'rb') as csvfile:
            myreader = csv.reader(csvfile, delimiter=',', quotechar='|')
            gtsrb_test_outputs = list(myreader)
        
        
        x_test_r = []
        y_test_r = []

        x_train_r = []
        y_train_r = []

        # train data
        outputs_correctness = []
        for i in xrange(len(gtsrb_outputs)):
            if i == 0:
                continue
            outputs_correctness.append(map(int, gtsrb_outputs[i][2].split(';')))

        fr = {}

        for j in xrange(len(outputs_correctness[0])):
            
            correctness = 0
            for i in xrange(len(gtsrb_outputs)):
                if i == 0:
                    continue
                correctness = correctness + outputs_correctness[i-1][j]
            #if correctness == 1:   3,31
            fail = (1001 - correctness)

            if fail in fr:
                fr[fail] = fr[fail] + 1
            else:
                fr[fail] = 0
        self.fr = fr
        # test data
        outputs_correctness = []
        for i in xrange(len(gtsrb_test_outputs)):
            if i == 0:
                continue
            outputs_correctness.append(map(int, gtsrb_test_outputs[i][2].split(';')))

       
        fr_test = {}
        for j in xrange(len(outputs_correctness[0])):
           
            correctness = 0
            for i in xrange(len(gtsrb_test_outputs)):
                if i == 0:
                    continue
                correctness = correctness + outputs_correctness[i-1][j]
            #if correctness == 1:
            fail = (1001 - correctness)

            if fail in fr_test:
                fr_test[fail] = fr_test[fail] + 1
            else:
                fr_test[fail] = 0
        self.fr_test = fr_test
    
    def initialize_dataset(self, threshold=8, threshold2=145):

        with open("gtsrb_layer_x_train_0.pkl", 'rb') as handle:
            layer_x_train = pickle.load(handle)
        with open("gtsrb_layer_x_test_0.pkl", 'rb') as handle:
            layer_x_test = pickle.load(handle)
        with open("gtsrb_layer_y_train_0.pkl", 'rb') as handle:
            y_train = pickle.load(handle)
        with open("gtsrb_layer_y_test_0.pkl", 'rb') as handle:
            y_test = pickle.load(handle)

        with open('gtsrb_exp14_3_grid0.csv', 'rb') as csvfile:
            myreader = csv.reader(csvfile, delimiter=',', quotechar='|')
            gtsrb_outputs = list(myreader)
        with open('gtsrb_exp14_test_3_grid0.csv', 'rb') as csvfile:
            myreader = csv.reader(csvfile, delimiter=',', quotechar='|')
            gtsrb_test_outputs = list(myreader)
        
        x_test_r = []
        y_test_r = []

        x_train_r = []
        y_train_r = []

        # train data
        outputs_correctness = []
        for i in xrange(len(gtsrb_outputs)):
            if i == 0:
                continue
            outputs_correctness.append(map(int, gtsrb_outputs[i][2].split(';')))

        
        for j in xrange(len(outputs_correctness[0])):
            
            correctness = 0
            for i in xrange(len(gtsrb_outputs)):
                if i == 0:
                    continue
                correctness = correctness + outputs_correctness[i-1][j]
            #if correctness == 1:   3,31
            #861 = 13*7*11
            fail = 1001 - correctness
            if fail < threshold:
                x_train_r.append(layer_x_train[j])
                y_train_r.append(1)
            elif fail >= threshold2:
                x_train_r.append(layer_x_train[j])
                y_train_r.append(0)

        # test data
        outputs_correctness = []
        for i in xrange(len(gtsrb_test_outputs)):
            if i == 0:
                continue
            outputs_correctness.append(map(int, gtsrb_test_outputs[i][2].split(';')))

       
       
        for j in xrange(len(outputs_correctness[0])):
           
            correctness = 0
            for i in xrange(len(gtsrb_test_outputs)):
                if i == 0:
                    continue
                correctness = correctness + outputs_correctness[i-1][j]
            fail = 1001 - correctness
            if fail < threshold:
                x_test_r.append(layer_x_test[j])
                y_test_r.append(1)
            elif fail >= threshold2:
                x_test_r.append(layer_x_test[j])
                y_test_r.append(0)
        
        self.x_train_r = np.array(x_train_r)
        self.y_train_r = y_train_r
        self.x_test_r = np.array(x_test_r)
        self.y_test_r = y_test_r
        print(np.array(x_train_r).shape)

        print("len of training data")
        print(len(x_train_r))
        print("training well")
        print(y_train_r.count(0))
        print("training pillar")
        print(y_train_r.count(1))
        print("len of test data")
        print(len(x_test_r))
        print("testing well")
        print(y_test_r.count(0))
        print("testing pillar")
        print(y_test_r.count(1))


    def fr_distribution(self):
        from scipy import stats
        import matplotlib.pyplot as plt

        y = []
        fr = self.fr
        for i in xrange(1001):
            if i in fr:
                y.append(fr[i])
            else:
                y.append(0)
        print("max" + str(max(fr)))
        x = xrange(1001)
        plt.plot(x,y,'-')
        
        plt.title("failure rate distribution")

        plt.xlabel('training data failure rate')
        plt.ylabel('number of images')
        plt.show()


        y = []
        fr = self.fr_test
        for i in xrange(1001):
            if i in fr:
                y.append(fr[i])
            else:
                y.append(0)
        x = xrange(1001)
        plt.plot(x,y,'-')
        
        plt.title("failure rate distribution")

        plt.xlabel('test data failure rate')
        plt.ylabel('number of images')
        plt.show()

    def train_model(self, action = "test", modelid=0, weights_file="pillar_well1.hdf5"):
        '''
        train or test a dnn model on mnist dataset
        action can have two values: "train", "test"
        '''

        model = Sequential()

        model.add(Dense(512, input_dim=512, kernel_initializer='normal', activation='relu'))
        model.add(Dense(128, kernel_initializer='normal', activation='relu'))
        model.add(Dense(64, kernel_initializer='normal', activation='relu'))
        model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
        # Compile model
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        
        #checkpoint = ModelCheckpoint(weights_file, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
        #callbacks_list = [checkpoint]
        '''
        seed = 1
        estimators = []
        estimators.append(('standardize', StandardScaler()))
        estimators.append(('mlp', KerasClassifier(build_fn=model, epochs=100, batch_size=5, verbose=0)))
        pipeline = Pipeline(estimators)
        kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
        results = cross_val_score(pipeline, self.x_train_r, self.y_train_r, cv=kfold)
        print("Smaller: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
        '''
        checkpoint = ModelCheckpoint(weights_file, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
        callbacks_list = [checkpoint]
        '''
        model.fit(self.x_train_r, self.y_train_r,
                      batch_size=5,
                      epochs=100, 
                      callbacks=callbacks_list,
                      verbose=1,
                      shuffle=True,
                      validation_data=(self.x_test_r, self.y_test_r))
        '''
        model.fit(self.x_train_r, self.y_train_r, validation_split=0.1, epochs=50, batch_size=10, callbacks=callbacks_list)
        score = model.evaluate(self.x_test_r, self.y_test_r, verbose=0)
        print('Test loss:', score[0])
        print('Test accuracy:', score[1])
        model.load_weights(weights_file)
        score = model.evaluate(self.x_test_r, self.y_test_r, verbose=0)
        print('Test loss:', score[0])
        print('Test accuracy:', score[1])


    def correlation(self, action = "test", modelid=0, weights_file="pillar_well1.hdf5"):
        
        with open("gtsrb_layer_x_train_0.pkl", 'rb') as handle:
            layer_x_train = pickle.load(handle)
        with open("gtsrb_layer_x_test_0.pkl", 'rb') as handle:
            layer_x_test = pickle.load(handle)
        with open("gtsrb_layer_y_train_0.pkl", 'rb') as handle:
            y_train = pickle.load(handle)
        with open("gtsrb_layer_y_test_0.pkl", 'rb') as handle:
            y_test = pickle.load(handle)

        with open('gtsrb_exp14_3_grid0.csv', 'rb') as csvfile:
            myreader = csv.reader(csvfile, delimiter=',', quotechar='|')
            gtsrb_outputs = list(myreader)
        with open('gtsrb_exp14_test_3_grid0.csv', 'rb') as csvfile:
            myreader = csv.reader(csvfile, delimiter=',', quotechar='|')
            gtsrb_test_outputs = list(myreader)

        model = Sequential()

        model.add(Dense(512, input_dim=512, kernel_initializer='normal', activation='relu'))
        model.add(Dense(128, kernel_initializer='normal', activation='relu'))
        model.add(Dense(64, kernel_initializer='normal', activation='relu'))
        model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
        # Compile model
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

        model.load_weights(weights_file)
        score = model.evaluate(self.x_test_r, self.y_test_r, verbose=0)
        print('Test loss:', score[0])
        print('Test accuracy:', score[1])

        outputs_correctness = []
        for i in xrange(len(gtsrb_test_outputs)):
            if i == 0:
                continue
            outputs_correctness.append(map(int, gtsrb_test_outputs[i][2].split(';')))

        
        fr = []
        new_x_test = []
        new_y_test = []
        for j in xrange(len(outputs_correctness[0])):

            correctness = 0
            for i in xrange(len(gtsrb_test_outputs)):
                if i == 0:
                    continue
                correctness = correctness + outputs_correctness[i-1][j]
            #if correctness == 31 or correctness == 1:
                #continue
            fr.append(1001 - correctness)
            new_x_test.append(layer_x_test[j])

        
        new_x_test = np.array(new_x_test)

        
        x_output = model.predict(new_x_test)
        from scipy import stats
        import matplotlib.pyplot as plt
        
        print(stats.spearmanr(fr, x_output))
        x = []
        for i in xrange(21):
            x.append([])

        for i in xrange(len(fr)):
            #if fr[i]%5 != 0:
                #continue
            #if fr[i]/10 == 15:
                #x[14].append(x_output[i])
            x[fr[i]/50].append(x_output[i])
        print(np.max(fr))
        temptick = list(xrange(1,22))
        
        plt.boxplot(x)
        plt.xticks(temptick, np.array(range(21))*50)
        plt.title("boxplot for 1001 failure rate")

        plt.xlabel('test data failure rate')
        plt.ylabel('model output')
        plt.show()

    def correlation_1(self, threshold=0.5):

        with open("gtsrb_layer_x_train_0.pkl", 'rb') as handle:
            layer_x_train = pickle.load(handle)
        with open("gtsrb_layer_x_test_0.pkl", 'rb') as handle:
            layer_x_test = pickle.load(handle)
        with open("gtsrb_layer_y_train_0.pkl", 'rb') as handle:
            y_train = pickle.load(handle)
        with open("gtsrb_layer_y_test_0.pkl", 'rb') as handle:
            y_test = pickle.load(handle)

        with open('gtsrb_exp14_3_grid0.csv', 'rb') as csvfile:
            myreader = csv.reader(csvfile, delimiter=',', quotechar='|')
            gtsrb_outputs = list(myreader)
        with open('gtsrb_exp14_test_3_grid0.csv', 'rb') as csvfile:
            myreader = csv.reader(csvfile, delimiter=',', quotechar='|')
            gtsrb_test_outputs = list(myreader)
        
        
        x_test_r = []
        y_test_r = []

        x_train_r = []
        y_train_r = []

        # train data
        outputs_correctness = []
        for i in xrange(len(gtsrb_outputs)):
            if i == 0:
                continue
            outputs_correctness.append(map(int, gtsrb_outputs[i][2].split(';')))

        fr = []
        for j in xrange(len(outputs_correctness[0])):
            
            correctness = 0
            for i in xrange(len(gtsrb_outputs)):
                if i == 0:
                    continue
                correctness = correctness + outputs_correctness[i-1][j]
            #if correctness == 1:   3,31
            fail = 1001 - correctness
            fr.append(fail*1.0/1001)
            

        class_well = {}
        for j in xrange(len(fr)):
            if fr[j] >= threshold:
                if y_train[j] in class_well:
                    class_well[y_train[j]] = class_well[y_train[j]] + 1
                else:
                    class_well[y_train[j]] = 1

        # test data
        outputs_correctness = []
        for i in xrange(len(gtsrb_test_outputs)):
            if i == 0:
                continue
            outputs_correctness.append(map(int, gtsrb_test_outputs[i][2].split(';')))

       
        class_fr = {}
        for j in xrange(len(outputs_correctness[0])):
           
            correctness = 0
            for i in xrange(len(gtsrb_test_outputs)):
                if i == 0:
                    continue
                correctness = correctness + outputs_correctness[i-1][j]
            fail = 1001 - correctness
            if y_test[j] in class_fr:
                class_fr[y_test[j]] = class_fr[y_test[j]] + fail
            else:
                class_fr[y_test[j]] = fail
        a = []
        b = []
        for i in xrange(10):
            a.append(class_well[i])
            b.append(class_fr[i]*1.0/1001/1000)
        print(a)
        print(b)
        from scipy import stats
        import matplotlib.pyplot as plt
        
        print(stats.spearmanr(a, b))
        
    def correlation_2(self, action = "test", modelid=0, weights_file="pillar_well1.hdf5"):
        
        with open("gtsrb_layer_x_train_0.pkl", 'rb') as handle:
            layer_x_train = pickle.load(handle)
        with open("gtsrb_layer_x_test_0.pkl", 'rb') as handle:
            layer_x_test = pickle.load(handle)
        with open("gtsrb_layer_y_train_0.pkl", 'rb') as handle:
            y_train = pickle.load(handle)
        with open("gtsrb_layer_y_test_0.pkl", 'rb') as handle:
            y_test = pickle.load(handle)

        with open('gtsrb_exp14_3_grid0.csv', 'rb') as csvfile:
            myreader = csv.reader(csvfile, delimiter=',', quotechar='|')
            gtsrb_outputs = list(myreader)
        with open('gtsrb_exp14_test_3_grid0.csv', 'rb') as csvfile:
            myreader = csv.reader(csvfile, delimiter=',', quotechar='|')
            gtsrb_test_outputs = list(myreader)

        model = Sequential()

        model.add(Dense(512, input_dim=512, kernel_initializer='normal', activation='relu'))
        model.add(Dense(128, kernel_initializer='normal', activation='relu'))
        model.add(Dense(64, kernel_initializer='normal', activation='relu'))
        model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
        # Compile model
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

        model.load_weights(weights_file)
        score = model.evaluate(self.x_test_r, self.y_test_r, verbose=0)
        print('Test loss:', score[0])
        print('Test accuracy:', score[1])

        outputs_correctness = []
        for i in xrange(len(gtsrb_test_outputs)):
            if i == 0:
                continue
            outputs_correctness.append(map(int, gtsrb_test_outputs[i][2].split(';')))

        exclusive = {}
        exclusive_list = map(int, gtsrb_test_outputs[16][2].split(';'))
        for j in xrange(len(exclusive_list)):        
            if exclusive_list[j]==0:
                exclusive[j] = 1
        fr = []
        new_x_test = []
        new_y_test = []
        for j in xrange(len(outputs_correctness[0])):
            if j in exclusive:
                continue
            correctness = 0
            for i in xrange(len(gtsrb_test_outputs)):
                if i == 0:
                    continue
                correctness = correctness + outputs_correctness[i-1][j]
            #if correctness == 31 or correctness == 1:
                #continue
            fr.append(31 - correctness)
            new_x_test.append(layer_x_test[j])

        
        new_x_test = np.array(new_x_test)

        
        x_output = model.predict(new_x_test)
        for i in xrange(len(x_output)):
            if x_output[i] >= 0.5:
                x_output[i] = 1
            else:
                x_output[i] = 0
        from scipy import stats
        import matplotlib.pyplot as plt
        '''
        print(stats.spearmanr(fr, x_output))
        x = []
        for i in xrange(31):
            x.append([])

        for i in xrange(len(fr)):
            x[fr[i]].append(x_output[i])
        print(np.max(fr))
        temptick = list(xrange(1,31))
        
        plt.boxplot(x)
        plt.xticks(temptick, list(xrange(31)))
        plt.title("boxplot for 30 failure rate")

        plt.xlabel('test data failure rate')
        plt.ylabel('model output')
        plt.show()
        '''
        y = {}
        y1 = [0]*31
        y0 = [0]*31
        for i in xrange(31):
            y[i] = []
        x = list(xrange(31))
        for i in xrange(len(fr)):
            y[fr[i]].append(x_output[i])
        for i in xrange(31):
            y1[i] = y[i].count(1)*1.0/len(y[i])
            y0[i] = y[i].count(0)*1.0/len(y[i])
        fig, ax = plt.subplots()
        
        ax.plot(x,y1,'g-',label='robustness pillar')
        ax.plot(x,y0,'r-',label='robustness hole')
        ax.legend(loc='best', shadow=True, fontsize='small')
        plt.title("Distribution of predicted robustness pillar and robustness hole")
        plt.xticks(np.arange(min(x), max(x)+1, 1.0))
        plt.yticks(np.arange(0, 1.1, 0.1))
        plt.xlabel('failure rate')
        plt.ylabel('percentage of test data with a specific failure rate')
        plt.show()


    def regression_model(self, weights_file="pillar_well_regression.hdf5"):
        with open("gtsrb_layer_x_train_0.pkl", 'rb') as handle:
            layer_x_train = pickle.load(handle)
        with open("gtsrb_layer_x_test_0.pkl", 'rb') as handle:
            layer_x_test = pickle.load(handle)
        with open("gtsrb_layer_y_train_0.pkl", 'rb') as handle:
            y_train = pickle.load(handle)
        with open("gtsrb_layer_y_test_0.pkl", 'rb') as handle:
            y_test = pickle.load(handle)

        with open('gtsrb_exp14_3_grid0.csv', 'rb') as csvfile:
            myreader = csv.reader(csvfile, delimiter=',', quotechar='|')
            gtsrb_outputs = list(myreader)
        with open('gtsrb_exp14_test_3_grid0.csv', 'rb') as csvfile:
            myreader = csv.reader(csvfile, delimiter=',', quotechar='|')
            gtsrb_test_outputs = list(myreader)

        
        x_test_r = []
        y_test_r = []

        x_train_r = []
        y_train_r = []

        # train data
        outputs_correctness = []
        for i in xrange(len(gtsrb_outputs)):
            if i == 0:
                continue
            outputs_correctness.append(map(int, gtsrb_outputs[i][2].split(';')))

        
        for j in xrange(len(outputs_correctness[0])):
            
            correctness = 0
            for i in xrange(len(gtsrb_outputs)):
                if i == 0:
                    continue
                correctness = correctness + outputs_correctness[i-1][j]
            #if correctness == 1:   3,31
            #861 = 13*7*11
            fail = 1001 - correctness

            x_train_r.append(layer_x_train[j])
            y_train_r.append(fail*1.0/1001)
          


        # test data
        fr = []
        outputs_correctness = []
        for i in xrange(len(gtsrb_test_outputs)):
            if i == 0:
                continue
            outputs_correctness.append(map(int, gtsrb_test_outputs[i][2].split(';')))

       
       
        for j in xrange(len(outputs_correctness[0])):
           
            correctness = 0
            for i in xrange(len(gtsrb_test_outputs)):
                if i == 0:
                    continue
                correctness = correctness + outputs_correctness[i-1][j]
            fail = 1001 - correctness
            fr.append(fail*1.0/1001)
            x_test_r.append(layer_x_test[j])
            y_test_r.append(fail*1.0/1001)

        self.x_train_r = np.array(x_train_r)
        self.y_train_r = y_train_r
        self.x_test_r = np.array(x_test_r)
        self.y_test_r = y_test_r
        print(np.array(x_train_r).shape)

        print("len of training data")
        print(len(x_train_r))
        
        print("len of test data")
        print(len(x_test_r))
        
        model = Sequential()

        model.add(Dense(512, input_dim=512, kernel_initializer='normal', activation='relu'))
        model.add(Dense(128, kernel_initializer='normal', activation='relu'))
        model.add(Dense(64, kernel_initializer='normal', activation='relu'))
        model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
        # Compile model
        model.compile(loss='mean_squared_error', optimizer='adam')
        
        #checkpoint = ModelCheckpoint(weights_file, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
        #callbacks_list = [checkpoint]
        '''
        seed = 1
        estimators = []
        estimators.append(('standardize', StandardScaler()))
        estimators.append(('mlp', KerasClassifier(build_fn=model, epochs=100, batch_size=5, verbose=0)))
        pipeline = Pipeline(estimators)
        kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
        results = cross_val_score(pipeline, self.x_train_r, self.y_train_r, cv=kfold)
        print("Smaller: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
        '''
        checkpoint = ModelCheckpoint(weights_file, verbose=1, save_best_only=True)
        callbacks_list = [checkpoint]
        '''
        model.fit(self.x_train_r, self.y_train_r,
                      batch_size=5,
                      epochs=100, 
                      callbacks=callbacks_list,
                      verbose=1,
                      shuffle=True,
                      validation_data=(self.x_test_r, self.y_test_r))
        '''
        model.fit(self.x_train_r, self.y_train_r, validation_split=0.1, shuffle=True, epochs=50, batch_size=10, callbacks=callbacks_list)
        score = model.evaluate(self.x_test_r, self.y_test_r, verbose=0)
        print('Test loss:', score)

        model.load_weights(weights_file)
        score = model.evaluate(self.x_test_r, self.y_test_r, verbose=0)
        print('Test loss:', score)


        x_output = model.predict(self.x_test_r)
        from scipy import stats
        import matplotlib.pyplot as plt
        
        print(stats.spearmanr(fr, x_output))

        well = []
        labelwell = []
        pillar = []
        labelpillar = []

        for i in xrange(len(x_output)):
            if x_output[i] < 0.2:
                well.append(i)
            if fr[i] < 0.2:
                labelwell.append(i)

            if x_output[i] > 0.8:
                pillar.append(i)
            if fr[i] > 0.8:
                labelpillar.append(i)
        inter = list(set(well) & set(labelwell))
        precision = len(inter) * 1.0/len(well)
        recall = len(inter) * 1.0/len(labelwell)
        print("well precision: " + str(precision))
        print("well recall: " + str(recall))

        inter = list(set(pillar) & set(labelpillar))
        precision = len(inter) * 1.0/len(pillar)
        recall = len(inter) * 1.0/len(labelpillar)
        print("pillar precision: " + str(precision))
        print("pillar recall: " + str(recall))

if __name__ == '__main__':
    md = robust_pillar()
    #md.fr_distribution()
    #md.initialize_dataset(threshold=300, threshold2=600)
    #md.train_model()
    
    #md.correlation()
    #md.correlation_1(threshold=0.5) # correlation between failure rate in each class and robustness well

    md.regression_model()
    #md.correlation_2(action="train")
    
