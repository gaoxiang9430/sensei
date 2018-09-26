'''
This program is designed to augment dataset
Author: Xiang Gao (xiang.gao@us.fujitsu.com)
Time: Sep, 24, 2018
'''

from transformation import *
import numpy as np
import random
import copy
from perturbator import Perturbator

class Augmentor:
    def __init__(self,):
        self.rotation_range = range(-30, 30)
        self.translate_range = range(-3, 4)
        self.shear_range = list(np.array(range(-20, 21))*1.0/100)
        self.trans_functions = {}
        self.trans_functions["rotate"] = image_rotation_cropped
        self.trans_functions["translate"] = image_translation_cropped
        self.trans_functions["shear"] = image_shear_cropped
        self.pt = Perturbator()

    def random_replace(self, x=None, y=None):
        length = range(len(x))
        for i in length:
            x[i] = self.pt.random_perturb_image(x[i])
        return x, y

    def random40_replace(self, x=None, y=None):
        self.pt.set_rotation_range(40)
        length = range(len(x))
        for i in length:
            x[i] = self.pt.random_perturb_image(x[i])
        self.pt.set_rotation_range(30)
        return x, y

        self.rotation_range = range(-30, 30)
        return x, y

    def worst_of_10(self, x=None, y=None):
        'randomly generate 10 perturbed example'
        length = range(len(x))
        x_10 = []
        for i in length:
            x_i_10 = []
            for j in range(10):
                img = copy.deepcopy(x[i])
                x_i_10.append(self.pt.random_perturb_image(img))
            x_10.append(x_i_10)
        return x_10, y

    def random_augment(self, x=None, y=None):
        return x, y

    def random_150_augment(self, x=None, y=None):
        return x, y

    def grid_augment(self, x=None, y=None):
        return x, y

