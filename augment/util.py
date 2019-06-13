"""
This program is designed to define supported strategies
Author: Xiang Gao (xiang.gao@us.fujitsu.com)
Time: Sep, 27, 2018
"""

from enum import Enum
import logging


class SAU(Enum):
    """supported_augmentation_strategy"""
    original = 1
    replace30 = 2
    replace40 = 4
    replace_worst_of_10 = 6
    ga_loss = 8
    ga_cov = 9

    @staticmethod
    def list():
        return list(map(lambda c: c.name, SAU))

    @staticmethod
    def get_name(strategy=None):
        for sau in SAU:
            if sau.name == strategy:
                return sau
        return None


class SAT(Enum):
    """supported_attack_strategy"""
    original = 1
    fix = 2
    random = 3
    random150 = 4
    grid1 = 5
    grid2 = 6
    rotate = 7
    translate = 8
    shear = 9

    @staticmethod
    def list():
        return list(map(lambda c: c.name, SAT))

    @staticmethod
    def get_name(strategy=None):
        for sat in SAT:
            if sat.name == strategy:
                return sat
        return None


class DATASET(Enum):
    """supported_dataset"""
    gtsrb = 1
    cifar10 = 2
    fashionmnist = 3
    svhn = 4
    imdb = 5
    utk = 6

    @staticmethod
    def list():
        return list(map(lambda c: c.name, DATASET))

    @staticmethod
    def get_name(dataset=None):
        for dat in DATASET:
            if dat.name == dataset:
                return dat
        return None

# create logger
logging.basicConfig()
logger = logging.getLogger('DeepAugment ')
logger.setLevel(logging.INFO)
