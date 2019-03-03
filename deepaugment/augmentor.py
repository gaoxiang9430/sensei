"""
This program is designed to augment dataset
Author: Xiang Gao (xiang.gao@us.fujitsu.com)
Time: Sep, 24, 2018
"""

import copy
from perturbator import Perturbator
import numpy as np
from config import ExperimentalConfig


class Augmenter:
    def __init__(self):
        config = ExperimentalConfig.gen_config()
        self.config = config
        self.pt = Perturbator()

        self.rotation_range = config.rotation_range
        self.translate_range = config.translate_range
        self.shear_range = config.shear_range

        self.zoom_range = config.zoom_range
        self.blur_range = config.blur_range
        self.brightness_range = config.brightness_range
        self.contrast_range = config.contrast_range

    """
    def uniform_random_replace(self, x=None, y=None):
        angle = random.choice(self.rotation_range)
        translation = random.choice(self.translate_range)
        shear = random.choice(self.shear_range)
        x, y = self.pt.fix_perturb(x, y, angle, translation, shear)
        return x, y
    """

    def random_replace(self, x=None, y=None):
        length = range(len(x))
        for i in length:
            x[i] = self.pt.random_perturb_image(x[i])
        return x, y

    def random_augment(self, x=None, y=None):
        augmented_x = []
        augmented_y = []
        length = range(len(x))
        for i in length:
            # add original data
            augmented_x.append(x[i])
            augmented_y.append(y[i])

            # add augmented data
            img = copy.deepcopy(x[i])
            temp = self.pt.random_perturb_image(img)
            augmented_x.append(temp)
            augmented_y.append(y[i])
        augmented_y = np.array(augmented_y)
        return augmented_x, augmented_y

    def random40_replace(self, x=None, y=None):
        self.pt.set_rotation_range(40)
        length = range(len(x))
        for i in length:
            x[i] = self.pt.random_perturb_image(x[i])
        self.pt.set_rotation_range(30)

        return x, y

    def worst_of_10(self, x=None, y=None):
        """randomly generate 10 perturbed examples for each image"""
        length = range(len(x))
        x_10 = []
        for j in range(10):
            x_i_10 = []
            for i in length:
                img = copy.deepcopy(x[i])
                x_i_10.append(self.pt.random_perturb_image(img))
            x_10.append(x_i_10)  # num_perturb * n
        return x_10, y

    def grid(self, x=None, y=None):
        x_grid = []
        for p1 in self.config.rotation_range[::20]:
            for p2 in self.config.translate_range[::2]:
                for p2_v in self.config.translate_range[::2]:
                    for p3 in self.config.shear_range[::15]:
                        if self.config.enable_filters:
                            for p4 in self.config.zoom_range[::10]:
                                for p5 in self.config.blur_range[::1]:
                                    for p6 in self.config.brightness_range[::16]:
                                        for p7 in self.config.contrast_range[::8]:
                                            temp_x_test = copy.deepcopy(x)
                                            x_perturbed_test, y_test = self.pt.fix_perturb(temp_x_test, y, p1, p2,
                                                                                           p2_v, p3, p4, p5, p6, p7)
                                            x_grid.append(x_perturbed_test)
                        else:
                            temp_x_test = copy.deepcopy(x)
                            x_perturbed_test, y_test = self.pt.fix_perturb(temp_x_test, y,
                                                                           p1, p2, p2_v, p3)
                            x_grid.append(x_perturbed_test)

        return x_grid, y

    def fix_perturb(self, x=None, y=None):
        length = range(len(x))
        for i in length:
            x[i] = self.pt.fix_perturb_img(x[i], 1)
        return x, y
