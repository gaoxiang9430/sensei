from transformation import Transformation
import copy
from config import global_config as config
from util import logger
from deepaugment.perturbator import Perturbator
import random
import numpy as np


class Item(object):
    def __init__(self, transformation_para=None, data=None, loss=0):
        self.data = data
        self.transformation_para = transformation_para
        self.loss = loss

    def __cmp__(self, other):
        return cmp(other.loss, self.loss)

    def __lt__(self, other):
        return other.loss < self.loss

    def compare_trans_para(self, other):
        return self.transformation_para.compare_paras(other.transformation_para)


class GridTransformation:
    def __init__(self, num_class):
        self.class_grid_transformation = dict()
        self.class_index = dict()
        trs = self.generate_transformations()
        for i in range(num_class):
            trs_temp = copy.deepcopy(trs)
            random.shuffle(trs_temp)
            self.class_grid_transformation[i] = trs_temp
            self.class_index[i] = 0

    @staticmethod
    def generate_transformations():
        trs = []
        for p1 in config.rotation_range[12::18]:           # 6 [-30, 30]
            for p2 in config.translate_range[1::2]:        # 1 [-3, 3]
                for p2_v in config.translate_range[1::2]:  # 1 [-3, 3]
                    for p3 in config.shear_range[8::12]:   # 4 [-20, 20]
                        if config.enable_filters:
                            for p4 in config.zoom_range[4::6]:                   # 2 [-10, 10]
                                for p5 in config.blur_range[::1]:                #
                                    for p6 in config.brightness_range[4::12]:    # 4 [-16, 16]
                                        for p7 in config.contrast_range[2::6]:   # 2 [-8, 8]
                                            tr = Transformation(p1, p2, p2_v, p3, p4, p5, p6, p7)
                                            trs.append(tr)
                        else:
                            tr = Transformation(p1, p2, p2_v, p3)
                            trs.append(tr)
        return trs

    def get_next_transformation(self, label):
        index = self.class_index[label]
        trs = self.class_grid_transformation[label]
        if index >= len(trs):
            random.shuffle(trs)
            self.class_grid_transformation[label] = trs
            index = 0
            self.class_index[label] = 0
        else:
            self.class_index[label] = index+1
        return trs[index]


class GASelect:

    def __init__(self, x_train=None, y_train=None, original_target=None):
        self.queue_set = []  # num_test * num_mutate
        self.queue_len = config.queue_len
        self.pt = Perturbator()
        self.original_target = original_target

        self.x_train = x_train
        self.y_train = y_train

        self.gt = GridTransformation(original_target.num_classes)
        # generate first population
        temp_x_original_train = copy.deepcopy(x_train)
        for i in range(len(temp_x_original_train)):
            label = np.argmax(y_train[i])
            q = list()
            q.append(Item(Transformation(), temp_x_original_train[i], 0))
            for j in range(9):
                img = copy.deepcopy(temp_x_original_train[i])
                tr = self.gt.get_next_transformation(label)
                mutated_img = self.pt.fix_perturb_img(img, *(tr.get_paras()))
                q.append(Item(tr, mutated_img, 0))
            #     angle = random.choice(config.rotation_range)
            #     translation = random.choice(config.translate_range)
            #     shear = random.choice(config.shear_range)
            #     # transformation based on filter
            #     if config.enable_filters:
            #         zoom = random.choice(config.zoom_range)
            #         blur = random.choice(config.blur_range)
            #         brightness = random.choice(config.brightness_range)
            #         contrast = random.choice(config.contrast_range)
            #         img = self.pt.fix_perturb_img(img, angle, translation, shear,
            #                                       zoom, blur, brightness, contrast)
            #         tr = Transformation(angle, translation, shear, zoom, blur, brightness, contrast)
            #     else:
            #         img = self.pt.fix_perturb_img(img, angle, translation, shear)
            #         tr = Transformation(angle, translation, shear)
            #     q.append(Item(tr, img, 0))
            self.queue_set.append(q)

    def generate_next_population(self, start=0, end=-1):
        if end == -1:
            end = len(self.queue_set)
        if self.original_target.__class__.__name__ == "Cifar10Model":
            for i in range(start, end):
                flip = random.choice([True, False])
                if flip:
                    self.x_train[i] = np.fliplr(self.x_train[i])

        r = random.choice(range(100)) / float(100)
        if r < config.prob_mutate:
            return self.mutate(start, end)
        else:
            return self.crossover(start, end)

    def mutate(self, start=0, end=-1):
        logger.debug("Using mutate operators")
        for i in range(start, end):
            q = self.queue_set[i]
            top_item = q[0]
            # q.remove(top_item)
            # if top_item.loss < 1e-3:
            #     del q[1:]
            #     for j in range(self.queue_len-1):
            #         img = copy.deepcopy(self.x_train[i])
            #         tr = self.gt.get_next_transformation(np.argmax(self.y_train[i]))
            #         mutated_img = self.pt.fix_perturb_img(img, *(tr.get_paras()))
            #         q.append(Item(tr, mutated_img, 0))
            # else:
            existing_trs = list(item.transformation_para for item in q)
            mutates = top_item.transformation_para.mutate(existing_trs, top_item.loss)
            q.remove(top_item)
            for j in range(len(mutates)):
                img = copy.deepcopy(self.x_train[i])
                mutated_img = self.pt.fix_perturb_img(img, *(mutates[j].get_paras()))
                q.append(Item(mutates[j], mutated_img, 0))
            # q.append(Item(Transformation(), copy.deepcopy(self.x_train[i]), 0))
        return self.get_all_data(start, end)

    def crossover(self, start=0, end=-1):
        logger.debug("Using crossover operators")
        for i in range(start, end):
            q = self.queue_set[i]
            if len(q) < 2:
                return self.mutate()
            top_item = q[0]
            # if top_item.loss < 1e-3:
            #     del q[1:]
            #     for j in range(self.queue_len-1):
            #         img = copy.deepcopy(self.x_train[i])
            #         tr = self.gt.get_next_transformation(np.argmax(self.y_train[i]))
            #         mutated_img = self.pt.fix_perturb_img(img, *(tr.get_paras()))
            #         q.append(Item(tr, mutated_img, 0))
            # else:
            top_item_2 = self.select_item(top_item, q[1:6])  # q[1]
            # q.remove(top_item)
            existing_trs = list(item.transformation_para for item in q)
            mutates = top_item.transformation_para.crossover(top_item_2.transformation_para, existing_trs,
                                                             top_item.loss)
            q.remove(top_item)
            for j in range(len(mutates)):
                img = copy.deepcopy(self.x_train[i])
                mutated_img = self.pt.fix_perturb_img(img, *(mutates[j].get_paras()))
                q.append(Item(mutates[j], mutated_img, 0))
            # q.append(Item(Transformation(), copy.deepcopy(self.x_train[i]), 0))
        return self.get_all_data(start, end)

    def get_all_data(self, start=0, end=-1):
        ret = []
        temp_queue_set = self.queue_set[start: end]
        logger.debug("the shape of queue set: " + str(np.array(temp_queue_set).shape))
        for i in range(len(temp_queue_set[0])):
            attr = list(item.data for item in np.array(temp_queue_set)[:, i])
            ret.append(attr)
        return ret  # num_mutate * num_test

    def fitness(self, loss, start=0, end=-1):
        if end == -1:
            end = len(self.queue_set)
        # logger.debug("update_queue - the shape of queue set: " + str(np.array(self.queue_set).shape))
        logger.debug("update_queue - the shape of loss: " + str(np.array(loss).shape))
        index = 0
        for i in range(start, end):         # for each test
            for j in range(len(self.queue_set[i])):  # for each mutation
                self.queue_set[i][j].loss = loss[j][index]
            index += 1
            self.queue_set[i].sort()
            # keep the top n and remove test with small loss
            self.queue_set[i] = self.queue_set[i][0:self.queue_len]
        if start == 0:
            for i in range(start, 20):  # for each test
                print("transformation para of " + str(i), self.queue_set[i][0].transformation_para.get_paras())
        logger.debug("update queue done")

    @ staticmethod
    def select_item(top_item, queue):
        maximal = 0
        top_item2 = queue[0]
        for item2 in queue:
            diff = top_item.compare_trans_para(item2)
            if diff > maximal:
                maximal = diff
                top_item2 = item2
        return top_item2

"""
if __name__ == '__main__':
    md = GtsrbModel("GTSRB")
    x_train_origin, y_train_origin = md.load_original_data('train')
    x_val_origin, y_val_origin = md.load_original_data('val')

    ga_selector = GASelect(x_train_origin, y_train_origin)
    ga_selector.mutate()
    import sys
    print sys.getsizeof(ga_selector)
"""

