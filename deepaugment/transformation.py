from config import global_config as config
import random
import numpy as np


class Transformation:

    def __init__(self, rotation=0, translate=0, translate_v=0, shear=0, zoom=1,
                 blur=0, brightness=0, contrast=1):
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

    @staticmethod
    def set_step(loss):
        if loss < 1e-3:
            config.translation_step['rotation'] = 12
            config.translation_step['shear'] = 0.04
        else:
            config.translation_step['rotation'] = 6
            config.translation_step['shear'] = 0.02

    def get_paras(self):
        return self.rotation, self.translate, self.translate_v, self.shear, \
               self.zoom, self.blur, self.brightness, self.contrast

    def flip(self):
        rotation = self.rotation * -1
        translate = self.translate * -1
        translate_v = self.translate_v * -1
        shear = self.shear * -1
        zoom = 2 - self.zoom
        blur = max(config.blur_range) - self.blur
        brightness = self.brightness * -1
        contrast = 2 - self.contrast
        return rotation, translate, translate_v, shear, zoom, blur, brightness, contrast

    def crossover(self, item2, existing_trs, loss):
        self.set_step(loss)

        mutated_params = []
        if config.enable_filters:
            choice = random.sample(range(1, 126), 6)  # 0000001 - 1111110
        else:
            choice = random.sample(range(1, 14), 6)   # 0001 - 1110

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
            if config.enable_filters:
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
        self.set_step(loss)

        mutated_params = []
        for i in range(6):
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

        if config.enable_filters:
            choice = random.sample(range(0, 7), 4)
        else:
            choice = random.sample(range(0, 4), 2)

        if 0 in choice:
            up_down_choice = random.choice([0, 1, 2])
            if up_down_choice == 0:
                rotation = self.fit_range(rotation + config.translation_step["rotation"],
                                          min(config.rotation_range), max(config.rotation_range))
            elif up_down_choice == 1:
                rotation = self.fit_range(rotation - config.translation_step["rotation"],
                                          min(config.rotation_range), max(config.rotation_range))
            else:
                rotation *= -1

        if 1 in choice or config.enable_optimize:
            up_down_choice = random.choice([0, 1, 2])
            if up_down_choice == 0:
                translate = self.fit_range(translate + config.translation_step["translate"],
                                           min(config.translate_range), max(config.translate_range))
            elif up_down_choice == 1:
                translate = self.fit_range(translate - config.translation_step["translate"],
                                           min(config.translate_range), max(config.translate_range))
            else:
                translate *= -1

        if 2 in choice or config.enable_optimize:
            up_down_choice = random.choice([0, 1, 2])
            if up_down_choice == 0:
                translate_v = self.fit_range(translate_v + config.translation_step["translate"],
                                             min(config.translate_range), max(config.translate_range))
            elif up_down_choice == 1:
                translate_v = self.fit_range(translate_v - config.translation_step["translate"],
                                             min(config.translate_range), max(config.translate_range))
            else:
                translate_v *= -1

        if 3 in choice:
            up_down_choice = random.choice([0, 1, 2])
            if up_down_choice == 0:
                shear = self.fit_range(shear + config.translation_step["shear"],
                                       min(config.shear_range), max(config.shear_range))
            elif up_down_choice == 1:
                shear = self.fit_range(shear - config.translation_step["shear"],
                                       min(config.shear_range), max(config.shear_range))
            else:
                shear *= -1

        if config.enable_filters:
            if 4 in choice:
                up_down_choice = random.choice([0, 1, 2])
                if up_down_choice == 0:
                    brightness = self.fit_range(brightness + config.translation_step["brightness"],
                                                min(config.brightness_range), max(config.brightness_range))
                elif up_down_choice == 1:
                    brightness = self.fit_range(brightness - config.translation_step["brightness"],
                                                min(config.brightness_range), max(config.brightness_range))
                else:
                    brightness *= -1

            if 5 in choice:
                up_down_choice = random.choice([0, 1, 2])
                if up_down_choice == 0:
                    contrast = self.fit_range(contrast + config.translation_step["contrast"],
                                              min(config.contrast_range), max(config.contrast_range))
                elif up_down_choice == 1:
                    contrast = self.fit_range(contrast - config.translation_step["contrast"],
                                              min(config.contrast_range), max(config.contrast_range))
                else:
                    contrast = 2 - contrast

            if 6 in choice:
                up_down_choice = random.choice([0, 1, 2])
                if up_down_choice == 0:
                    zoom = self.fit_range(zoom + config.translation_step["zoom"],
                                          min(config.zoom_range), max(config.zoom_range))
                elif up_down_choice == 1:
                    zoom = self.fit_range(zoom - config.translation_step["zoom"],
                                          min(config.zoom_range), max(config.zoom_range))
                else:
                    zoom = 2 - zoom

        return Transformation(rotation, translate, translate_v, shear, zoom, blur, brightness, contrast)
