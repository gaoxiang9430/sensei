import numpy as np
from util import logger
import pickle
import os.path
import commands


class Config:
    config = None
    queue_len = 4  # config the size of queue for genetic algorithm
    prob_mutate = 0.6  # mutate probability

    num_processor = 56  # the number of process (for multiprocessing)
    coverage_threshold = 0.5  # differential coverage threshold
    robust_threshold = 1e-2
    robust_basedon_acc = True

    # to enable translation based on filter
    enable_filters = True  # zoom, blur, brightness, contrast
    """ define translation range """
    rotation_range = range(-30, 31)  # [-30, 30, 1]       - 60
    # rotation_range is set to 15 for cifar 10
    # rotation_range = range(-15, 15)                          # [-15, 15, 1]       - 30
    translate_range = range(-3, 4)  # [-3, 3, 1]         - 6
    shear_range = list(np.array(range(-20, 21)) * 1.0 / 200)  # [-0.1, 0.1, 0.005] - 40
    """ define filter translation range """
    # zoom_range = range(1, 2)
    zoom_range = list(np.array(range(90, 111)) * 1.0 / 100)  # [0.9, 1.1, 0.01]   - 20
    blur_range = range(0, 1)  # [0, 3, 1]          - 3
    brightness_range = list(np.array(range(-16, 17)) * 2)  # [-32, 32, 2]       - 32
    contrast_range = list(np.array(range(32, 49)) * 1.0 / 40)  # [0.8, 1.2, 0.025]  - 16

    # mutate step in genetic algorithm
    translation_step = {"rotation": 6, "translate": 1, "shear": 0.02, "zoom": 0.02,
                        "blur": 1, "brightness": 4, "contrast": 0.05}

    enable_optimize = False
    epoch_level_augment = False

    def print_config(self):
        logger.info("=============== global config ===============")
        logger.info("queue length : " + str(self.queue_len))
        logger.info("mutate probability : " + str(self.prob_mutate))
        logger.info("the number of process (for multiprocessing) : " + str(self.num_processor))
        logger.info("coverage differential threshold : " + str(self.coverage_threshold))
        logger.info("enable transformation based on filter : " + str(self.enable_filters))
        logger.info("enable optimize : " + str(self.enable_optimize))
        logger.info("robust_threshold : " + str(self.robust_threshold))

        logger.info("=============== translation config ===============")
        logger.info("rotation range : " + str(self.rotation_range))
        logger.info("translate range : " + str(self.translate_range))
        logger.info("shear range : " + str(self.shear_range))
        if self.enable_filters:
            logger.info("zoom range : " + str(self.zoom_range))
            logger.info("blur range : " + str(self.blur_range))
            logger.info("brightness range : " + str(self.brightness_range))
            logger.info("contrast range : " + str(self.contrast_range))
        logger.info("mutate step (for genetic algorithm) : " + str(self.translation_step))

        logger.info("=============== Training Start ===============")

    def __init__(self):
        pass


class ExperimentalConfig:
    config = None
    system_id = commands.getstatusoutput("ifconfig | grep eno1 | awk '{print $NF}' | sed 's/://g'")[1]

    @staticmethod
    def gen_config():
        # get a system unique id
        if ExperimentalConfig.config is not None:
            return ExperimentalConfig.config
        elif os.path.isfile('/tmp/config' + ExperimentalConfig.system_id + '.pkl'):
            with open('/tmp/config' + ExperimentalConfig.system_id + '.pkl', 'rb') as inputs:
                config = pickle.load(inputs)
        else:
            config = Config()
            ExperimentalConfig.save_config(config)
        return config

    @staticmethod
    def save_config(config):
        # get a system unique id
        system_id = commands.getstatusoutput("ifconfig | grep eno1 | awk '{print $NF}' | sed 's/://g'")[1]

        print("config optimize: " + str(config.enable_optimize))
        print("config filter: " + str(config.enable_filters))
        with open('/tmp/config' + system_id + '.pkl', 'wb') as output:
            pickle.dump(config, output, pickle.HIGHEST_PROTOCOL)
