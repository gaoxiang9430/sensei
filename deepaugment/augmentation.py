'''
This program is designed to augment dataset
Author: Xiang Gao (xiang.gao@us.fujitsu.com)
Time: Sep, 24, 2018
'''

from transformation import *
import numpy as np
import random

class Augmentor:
    def __init__(self,):
        self.rotation_range = range(-30, 30)
        self.translate_range = range(-3, 4)
        self.shear_range = list(np.array(range(-20, 21))*1.0/100)
        self.trans_functions = {}
        self.trans_functions["rotate"] = image_rotation_cropped
        self.trans_functions["translate"] = image_translation_cropped
        self.trans_functions["shear"] = image_shear_cropped

    def random_augment(self, x=None, y=None):
        for i in range(len(x)):
            angle = random.choice(self.rotation_range)
            translation = random.choice(self.translate_range)
            shear = random.choice(self.shear_range)
            x[i] = self.trans_functions["rotate"](x[i], angle)
            x[i] = self.trans_functions["translate"](x[i], translation)
            x[i] = self.trans_functions["shear"](x[i], shear)

        return x, y

    def random_150_augment(self):
        return 0

    def grid_augment(selg):
        return 0

    def random_perturb(self, x=None, y=None):
        for i in range(len(x)):
            angle = random.choice(self.rotation_range)
            x[i]=self.trans_functions["rotate"](x[i], angle)

        return x, y

    def fix_perturb(self, x=None, y=None):
        for i in range(len(x)):
            x[i]=self.trans_functions["rotate"](x[i], 15)

        return x, y
