'''
This program is designed to attach a given models
Author: Xiang Gao (xiang.gao@us.fujitsu.com)
Time: Sep, 21, 2018
'''

from dataset.gtsrb.train import gtsrb_model
from augmentation import *
import numpy as np

class Attack_Model:

    def __init__(self, target=None):
        self.supported_strategy =  ['random', 'random_150', 'grid']
        self.target = target

    def attack(self, strategy="random", _model=None):
        x_test, y_test = self.target.load_original_test_data()
        aug = Augmentor()
        if strategy == "random":
            x_test, y_test = aug.random_augment(x_test, y_test)
        else:
            x_test, y_test = aug.fix_perturb(x_test, y_test)

        model = target.load_model(_model[0], _model[1])
        self.target.test_dnn_model(model, self.target.preprocess_original_imgs(x_test), y_test)

if __name__ == '__main__':
    target = gtsrb_model(source_dir='GTSRB')
    #_model = [0, "models/gtsrb.oxford.model0.hdf5"]
    _model = [0, "models/gtsrb.oxford.augmented_model0.hdf5"]
    _model2 = [0, "models/gtsrb.oxford.augmented_model.hdf5"]
    am = Attack_Model(target)
    am.attack('fix', _model)
    am.attack("random", _model)

    am.attack('fix', _model2)
    am.attack("random", _model2)
