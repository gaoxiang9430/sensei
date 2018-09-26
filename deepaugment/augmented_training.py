'''
This program is designed to train a model based on augmented dataset
Author: Xiang Gao (xiang.gao@us.fujitsu.com)
Time: Sep, 21, 2018
'''

from dataset.gtsrb.train import gtsrb_model
from augmentor import *
import keras
from sklearn.metrics import roc_auc_score
import numpy as np
import copy

supported_strategy =  ["original", 'replace30', 'replace40', 'replace_worst_of_10','augment_random', 'augment_worst_of_10']

class Data_Generator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, original_model=None, x_original_train=None, y_train=None, batch_size=32, strategy="replace30"):
        self.au = Augmentor()
        self.batch_size = batch_size
        self.original_model = original_model
        self.x_original_train = x_original_train
        self.strategy = strategy

        temp_x_original_train = copy.deepcopy(self.x_original_train)
        self.x_train = original_model.preprocess_original_imgs(temp_x_original_train)
        self.y_train = y_train

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.ceil(len(self.x_train) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data according to the batch index'
        start = index*self.batch_size
        end = (index+1)*self.batch_size
        if end > len(self.x_train):
            end = len(self.x_train)
        x = self.x_train[start:end]
        y = self.y_train[start:end]

        return x, y

    def on_epoch_end(self):
        'perturb the training sets after each epoch'
        if self.strategy == "original":
            return
        elif self.strategy == 'replace30':
            del self.x_train
            temp_x_original_train = copy.deepcopy(self.x_original_train)
            self.x_train, self.y_train = self.au.random_replace(temp_x_original_train, self.y_train)
            self.x_train = self.original_model.preprocess_original_imgs(self.x_train)
        elif self.strategy == 'replace40':
            del self.x_train
            temp_x_original_train = copy.deepcopy(self.x_original_train)
            self.x_train, self.y_train = self.au.random40_replace(temp_x_original_train, self.y_train)
            self.x_train = self.original_model.preprocess_original_imgs(self.x_train)
        elif self.strategy == 'replace_worst_of_10':
            del self.x_train
            temp_x_original_train = copy.deepcopy(self.x_original_train)
            x_train_10, self.y_train = self.au.worst_of_10(temp_x_original_train, self.y_train)
            #TODO: evaluate 10 example, and update self.x_train
            self.x_train = self.original_model.preprocess_original_imgs(self.x_train)
        else:
            raise Exception('unsupported augment strategy')
        print(" Perturbation Done!!!")

class Augmented_Model:

    def __init__(self, target=None):
        self.target = target

    def train(self, strategy="replace30", _model=None):
        x_train, y_train = target.load_original_data('train')
        x_val, y_val = target.load_original_data('val')

        data_generator = Data_Generator(self.target, x_train, y_train, 32, strategy)

        model=target.train_dnn_model(model_id=_model[0],weights_file=_model[1],x_train=None, y_train=None, x_val=target.preprocess_original_imgs(x_val), y_val=y_val, data_generator=data_generator)
        #model=target.train_dnn_model(model_id=_model[0],weights_file=_model[1],x_train=target.preprocess_original_imgs(x_train), y_train=y_train, x_val=target.preprocess_original_imgs(x_val), y_val=y_val, data_generator=data_generator)
   
    def test(self, _model=None):
        x_test, y_test = target.load_original_test_data()
        model = target.load_model(_model[0], _model[1])
        target.test_dnn_model(model, target.preprocess_original_imgs(x_test), y_test)

if __name__ == '__main__':
    target = gtsrb_model(source_dir='GTSRB')
    atm = Augmented_Model(target)

    #_model30 = [0, "models/gtsrb.oxford.replace30_model.hdf5"]
    #atm.train("replace30", _model30)
    #atm.test(_model30)

    _model40 = [0, "models/gtsrb.oxford.replace40_model.hdf5"]
    atm.train("replace40", _model40)
    atm.test(_model40)
