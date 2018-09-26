'''
This program is designed to perturbe dataset
Author: Xiang Gao (xiang.gao@us.fujitsu.com)
Time: Sep, 25, 2018
'''

from transformation import *
import numpy as np
import random

class Perturbator:
    def __init__(self):
        self.rotation_range = range(-30, 30)
        self.translate_range = range(-3, 4)
        self.shear_range = list(np.array(range(-20, 21))*1.0/100)
        self.trans_functions = {}
        self.trans_functions["rotate"] = image_rotation_cropped
        self.trans_functions["translate"] = image_translation_cropped
        self.trans_functions["shear"] = image_shear_cropped

    def set_rotation_range(self, _range=30):
        self.rotation_range = range(_range*-1, _range)

    def random_rotate_perturb(self, x=None, y=None):
        for i in range(len(x)):
            angle = random.choice(self.rotation_range)
            x[i] = self.trans_functions["rotate"](x[i], angle)
        return x, y


    def random_perturb(self, x=None, y=None):
        length = range(len(x))
        for i in length:
            x[i] = self.random_perturb_image(x[i])
        return x, y

    def random_perturb_image(self, img=None):
        'randomly perturb one image'
        angle = random.choice(self.rotation_range)
        translation = random.choice(self.translate_range)
        shear = random.choice(self.shear_range)
        img = self.trans_functions["rotate"](img, angle)
        img = self.trans_functions["translate"](img, translation)
        img = self.trans_functions["shear"](img, shear)
        return img


    def fix_perturb(self, x=None, y=None):
        length = range(len(x))
        for i in length:
            x[i]=self.trans_functions["rotate"](x[i], 15)
            x[i]=self.trans_functions["translate"](x[i], 2)
            x[i]=self.trans_functions["shear"](x[i], 0.1)
        return x, y

    def grid_perturb(self, x=None, y=None):
        return                    
