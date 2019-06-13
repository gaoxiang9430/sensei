import random
import numpy as np
from config import ExperimentalConfig


class Transformation:

    def __init__(self, rotation=0, translate=0, translate_v=0, shear=0, zoom=1,
                 blur=0, brightness=0, contrast=1):
        self.config = ExperimentalConfig.gen_config()
        self.rotation = rotation
        self.translate = translate
        self.translate_v = translate_v
        self.shear = shear
        self.zoom = zoom
        self.blur = blur
        self.brightness = brightness
        self.contrast = contrast

    def __eq__(self, other):
        return self.rotation == other.rotation\
               and self.translate == other.translate \
               and self.translate_v == other.translate_v \
               and self.shear == other.shear\
               and self.zoom == other.zoom\
               and self.blur == other.blur\
               and self.brightness == other.brightness\
               and self.contrast == other.contrast

    def compare_paras(self, other):
        diff = [self.rotation == other.rotation,
                self.translate == other.translate,
                self.translate_v == other.translate_v,
                self.shear == other.shear,
                self.zoom == other.zoom,
                self.blur == other.blur,
                self.brightness == other.brightness,
                self.contrast == other.contrast]
        return len(diff) - np.array(diff, dtype=int).sum()

    @staticmethod
    def fit_range(x, x_min, x_max):
        if x > x_max:
            x = x_max
        if x < x_min:
            x = x_min
        return x

    def get_paras(self):
        return self.rotation, self.translate, self.translate_v, self.shear, \
               self.zoom, self.blur, self.brightness, self.contrast

    def flip(self):
        rotation = self.rotation * -1
        translate = self.translate * -1
        translate_v = self.translate_v * -1
        shear = self.shear * -1
        zoom = 2 - self.zoom
        blur = max(self.config.blur_range) - self.blur
        brightness = self.brightness * -1
        contrast = 2 - self.contrast
        return rotation, translate, translate_v, shear, zoom, blur, brightness, contrast

    def crossover(self, item2, existing_trs, loss):
        if loss < 1e-3:
            self.config.translation_step['rotation'] = 12
            self.config.translation_step['shear'] = 0.04
        else:
            self.config.translation_step['rotation'] = 6
            self.config.translation_step['shear'] = 0.02

        mutated_params = []
        if self.config.enable_filters:
            choice = random.sample(range(1, 126), self.config.popsize)  # 0000001 - 1111110
        else:
            if self.config.popsize > 13:
                choice = random.sample(range(1, 14), 13)   # 0001 - 1110
                for i in range(0, self.config.popsize-13):
                    choice.append(0)
            else:
                choice = random.sample(range(1, 14), self.config.popsize)   # 0001 - 1110

        for i in choice:
            rotation, translate, translate_v, shear, zoom, blur, brightness, contrast = self.get_paras()
            ids = format(i, '#010b')
            if ids[-1] == '1':
                rotation = item2.rotation
            if ids[-2] == '1':
                translate = item2.translate
            if ids[-3] == '1':
                translate_v = item2.translate_v
            if ids[-4] == '1':
                shear = item2.shear
            if self.config.enable_filters:
                if ids[-5] == '1':
                    brightness = item2.brightness
                if ids[-6] == '1':
                    contrast = item2.contrast
                if ids[-7] == '1':
                    zoom = item2.zoom
                # if ids[-7] == '1':
                #     blur = item2.blur
            tr = Transformation(rotation, translate, translate_v, shear, zoom, blur, brightness, contrast)
            if tr in existing_trs:
                tr = self.mutate_node(tr, existing_trs)
            existing_trs.append(tr)
            mutated_params.append(tr)

        return mutated_params

    def mutate(self, existing_trs, loss):
        if loss < 1e-3:
            self.config.translation_step['rotation'] = 12
            self.config.translation_step['shear'] = 0.04
        else:
            self.config.translation_step['rotation'] = 6
            self.config.translation_step['shear'] = 0.02

        mutated_params = []
        for i in range(self.config.popsize):
            tr = self.mutate_node(self, existing_trs)
            mutated_params.append(tr)
            existing_trs.append(tr)

        return mutated_params

    def mutate_node(self, cur_tr, existing_trs):
        for i in range(10):
            new_tr = self.generate_mutated_node(cur_tr)
            if new_tr not in existing_trs:
                return new_tr
        return cur_tr

    def generate_mutated_node(self, cur_tr):
        rotation, translate, translate_v, shear, zoom, blur, brightness, contrast = cur_tr.get_paras()

        if self.config.enable_filters:
            choice = random.sample(range(0, 7), 4)
        else:
            choice = random.sample(range(0, 4), 2)

        if 0 in choice:
            up_down_choice = random.choice([0, 1, 2])
            if up_down_choice == 0:
                rotation = self.fit_range(rotation + self.config.translation_step["rotation"],
                                          min(self.config.rotation_range), max(self.config.rotation_range))
            elif up_down_choice == 1:
                rotation = self.fit_range(rotation - self.config.translation_step["rotation"],
                                          min(self.config.rotation_range), max(self.config.rotation_range))
            else:
                rotation *= -1

        if 1 in choice or self.config.enable_optimize:
            up_down_choice = random.choice([0, 1, 2])
            if up_down_choice == 0:
                translate = self.fit_range(translate + self.config.translation_step["translate"],
                                           min(self.config.translate_range), max(self.config.translate_range))
            elif up_down_choice == 1:
                translate = self.fit_range(translate - self.config.translation_step["translate"],
                                           min(self.config.translate_range), max(self.config.translate_range))
            else:
                translate *= -1

        if 2 in choice or self.config.enable_optimize:
            up_down_choice = random.choice([0, 1, 2])
            if up_down_choice == 0:
                translate_v = self.fit_range(translate_v + self.config.translation_step["translate"],
                                             min(self.config.translate_range), max(self.config.translate_range))
            elif up_down_choice == 1:
                translate_v = self.fit_range(translate_v - self.config.translation_step["translate"],
                                             min(self.config.translate_range), max(self.config.translate_range))
            else:
                translate_v *= -1

        if 3 in choice:
            up_down_choice = random.choice([0, 1, 2])
            if up_down_choice == 0:
                shear = self.fit_range(shear + self.config.translation_step["shear"],
                                       min(self.config.shear_range), max(self.config.shear_range))
            elif up_down_choice == 1:
                shear = self.fit_range(shear - self.config.translation_step["shear"],
                                       min(self.config.shear_range), max(self.config.shear_range))
            else:
                shear *= -1

        if self.config.enable_filters:
            if 4 in choice:
                up_down_choice = random.choice([0, 1, 2])
                if up_down_choice == 0:
                    brightness = self.fit_range(brightness + self.config.translation_step["brightness"],
                                                min(self.config.brightness_range), max(self.config.brightness_range))
                elif up_down_choice == 1:
                    brightness = self.fit_range(brightness - self.config.translation_step["brightness"],
                                                min(self.config.brightness_range), max(self.config.brightness_range))
                else:
                    brightness *= -1

            if 5 in choice:
                up_down_choice = random.choice([0, 1, 2])
                if up_down_choice == 0:
                    contrast = self.fit_range(contrast + self.config.translation_step["contrast"],
                                              min(self.config.contrast_range), max(self.config.contrast_range))
                elif up_down_choice == 1:
                    contrast = self.fit_range(contrast - self.config.translation_step["contrast"],
                                              min(self.config.contrast_range), max(self.config.contrast_range))
                else:
                    contrast = 2 - contrast

            if 6 in choice:
                up_down_choice = random.choice([0, 1, 2])
                if up_down_choice == 0:
                    zoom = self.fit_range(zoom + self.config.translation_step["zoom"],
                                          min(self.config.zoom_range), max(self.config.zoom_range))
                elif up_down_choice == 1:
                    zoom = self.fit_range(zoom - self.config.translation_step["zoom"],
                                          min(self.config.zoom_range), max(self.config.zoom_range))
                else:
                    zoom = 2 - zoom

        return Transformation(rotation, translate, translate_v, shear, zoom, blur, brightness, contrast)
