"""
This program is designed to perturbe dataset
Author: Xiang Gao (xiang.gao@us.fujitsu.com)
Time: Sep, 25, 2018
"""

import random

from config import ExperimentalConfig
from libs.spacial_transformation import *


class Perturbator:
    def __init__(self):
        config = ExperimentalConfig.gen_config()
        self.rotation_range = config.rotation_range
        self.translate_range = config.translate_range
        self.shear_range = config.shear_range

        self.zoom_range = config.zoom_range
        self.blur_range = config.blur_range
        self.brightness_range = config.brightness_range
        self.contrast_range = config.contrast_range

        self.trans_functions = dict()
        self.trans_functions["rotate"] = rotate_image
        self.trans_functions["translate"] = image_translation_cropped
        self.trans_functions["shear"] = image_shear_cropped

        # the function of filters
        self.trans_functions["zoom"] = image_zoom
        self.trans_functions["blur"] = image_blur
        self.trans_functions["contrast"] = image_contrast
        self.trans_functions["brightness"] = image_brightness

        self.enable_filters = config.enable_filters

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
        """randomly perturb one image"""
        # spacial transformation
        angle = random.choice(self.rotation_range)
        translation = random.choice(self.translate_range)
        translation_v = random.choice(self.translate_range)
        shear = random.choice(self.shear_range)
        zoom = 1
        blur = 0
        brightness = 0
        contrast = 1
        # transformation based on filter
        if self.enable_filters:
            zoom = random.choice(self.zoom_range)
            blur = random.choice(self.blur_range)
            brightness = random.choice(self.brightness_range)
            contrast = random.choice(self.contrast_range)
        img = self.fix_perturb_img(img, angle, translation, translation_v, shear, zoom, blur, brightness, contrast)
        return img

    def fix_perturb(self, x=None, y=None, angle=15, translation=2, translation_v=0, shear=0.1,
                    zoom=1, blur=0,  brightness=0, contrast=1):
        length = range(len(x))
        for i in length:
            x[i] = self.fix_perturb_img(x[i], angle, translation, translation_v,
                                        shear, zoom, blur, brightness, contrast)
        return x, y

    def fix_perturb_img(self, img, angle=15, translation=0, translation_v=0, shear=0.1,
                        zoom=1, blur=0, brightness=0, contrast=1):
        # img = img[:, :, ::-1]
        # translation_v = random.choice(self.translate_range)
        img = self.trans_functions["rotate"](img, angle)
        img = self.trans_functions["translate"](img, translation, translation_v)
        img = self.trans_functions["shear"](img, shear)
        if self.enable_filters:
            img = self.trans_functions["zoom"](img, zoom)
            img = self.trans_functions["blur"](img, blur)
            img = self.trans_functions["brightness"](img, brightness)
            img = self.trans_functions["contrast"](img, contrast)
        # img = img[:, :, ::-1]
        return img
