'''
This program is designed to train a model based on augmented dataset
Author: Xiang Gao (xiang.gao@us.fujitsu.com)
Time: Sep, 21, 2018
'''

from dataset.gtsrb.train import gtsrb_model
from augmentation import *
import keras
from sklearn.metrics import roc_auc_score
import numpy as np
import copy

class Augment30_Generator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, original_model=None, x_original_train=None, y_train=None, batch_size=32):
        self.batch_size = batch_size
        self.original_model = original_model
        self.x_original_train = x_original_train

        temp_x_original_train = copy.deepcopy(self.x_original_train)
        self.x_train = original_model.preprocess_original_imgs(temp_x_original_train)
        self.y_train = y_train

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.ceil(len(self.x_train) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        start = index*self.batch_size
        end = (index+1)*self.batch_size
        if end > len(self.x_train):
            end = len(self.x_train)
        x = self.x_train[start:end]
        y = self.y_train[start:end]

        return x, y

    def on_epoch_end(self):
        'perturb the training sets after each epoch'
        au = Augmentor()
        del self.x_train
        temp_x_original_train = copy.deepcopy(self.x_original_train)
        #self.x_train, self.y_train = self.original_model.load_original_data('train')
        #self.x_train = au.random_perturb(temp_x_original_train)
        self.x_train, self.y_train = au.random_perturb(temp_x_original_train, self.y_train)
        self.x_train = self.original_model.preprocess_original_imgs(self.x_train)
        print("Perturbation Done!!!")

class Augmented_Model:

    supported_strategy =  ['augment30', 'augment40']

    def __init__(self, target=None):
        self.target = target

    def train(self, strategy="random", _model=None):
        x_train, y_train = target.load_original_data('train')
        x_val, y_val = target.load_original_data('val')

        data_generator = Augment30_Generator(self.target, x_train, y_train)

        model=target.train_dnn_model(model_id=_model[0],weights_file=_model[1],x_train=None, y_train=None, x_val=target.preprocess_original_imgs(x_val), y_val=y_val, data_generator=data_generator)
        #model=target.train_dnn_model(model_id=_model[0],weights_file=_model[1],x_train=target.preprocess_original_imgs(x_train), y_train=y_train, x_val=target.preprocess_original_imgs(x_val), y_val=y_val, data_generator=data_generator)
   
    def test(self, _model=None):
        x_test, y_test = target.load_original_test_data()
        model = target.load_model(_model[0], _model[1])
        target.test_dnn_model(model, target.preprocess_original_imgs(x_test), y_test)

if __name__ == '__main__':
    target = gtsrb_model(source_dir='GTSRB')
    _model = [0, "models/gtsrb.oxford.augmented_model.hdf5"]
    
    atm = Augmented_Model(target)
    #atm.train("augmendt30", _model)
    atm.test(_model)
