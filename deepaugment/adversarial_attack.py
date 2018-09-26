'''
This program is designed to attach a given models
Author: Xiang Gao (xiang.gao@us.fujitsu.com)
Time: Sep, 21, 2018
'''

from dataset.gtsrb.train import gtsrb_model
from perturbator import Perturbator
import numpy as np

class Attack_Model:

    def __init__(self, target=None):
        self.supported_strategy =  ['random', 'random_150', 'grid']
        self.target = target

    def attack(self, strategy="random", _model=None, print_label=""):
        x_test, y_test = self.target.load_original_test_data()
        pt = Perturbator()
        if strategy == "random":
            x_test, y_test = pt.random_perturb(x_test, y_test)
        else:
            x_test, y_test = pt.fix_perturb(x_test, y_test)

        model = target.load_model(_model[0], _model[1])
        self.target.test_dnn_model(model, self.target.preprocess_original_imgs(x_test), y_test, print_label)

if __name__ == '__main__':
    target = gtsrb_model(source_dir='GTSRB')
    #_model = [0, "models/gtsrb.oxford.model0.hdf5"]
    _model = [0, "models/gtsrb.oxford.model.hdf5"]
    _model30 = [0, "models/gtsrb.oxford.replace30_model.hdf5"]
    _model40 = [0, "models/gtsrb.oxford.replace40_model.hdf5"]

    am = Attack_Model(target)
    am.attack('fix', _model, "fix(15) attack original oxford model")
    am.attack("random", _model, "random attack original oxford model")

    am.attack('fix', _model30, "fix(15) attack replace30 oxford model")
    am.attack("random", _model30, "random attack replace30 oxford model")

    am.attack('fix', _model40, "fix(15) attack replace40 oxford model")
    am.attack("random", _model40, "random attack replace40 oxford model")

