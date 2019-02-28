import numpy as np
from util import logger


class ExperimentalConfig:
    queue_len = 4              # config the size of queue for genetic algorithm
    prob_mutate = 0.6          # mutate probability

    num_processor = 56         # the number of process (for multiprocessing)
    coverage_threshold = 0.5   # differential coverage threshold

    # to enable translation based on filter
    enable_filters = False     # zoom, blur, brightness, contrast
    """ define translation range """
    rotation_range = range(-30, 31)                            # [-30, 30, 1]       - 60
    # rotation_range is set to 15 for cifar 10
    # rotation_range = range(-15, 15)                          # [-15, 15, 1]       - 30
    translate_range = range(-3, 4)                             # [-3, 3, 1]         - 6
    shear_range = list(np.array(range(-20, 21)) * 1.0 / 200)   # [-0.1, 0.1, 0.005] - 40
    """ define filter translation range """
    # zoom_range = range(1, 2)
    zoom_range = list(np.array(range(90, 111)) * 1.0 / 100)    # [0.9, 1.1, 0.01]   - 20
    blur_range = range(0, 1)                                   # [0, 3, 1]          - 3
    brightness_range = list(np.array(range(-16, 17)) * 2)      # [-32, 32, 2]       - 32
    contrast_range = list(np.array(range(32, 49)) * 1.0 / 40)  # [0.8, 1.2, 0.025]  - 16

    # mutate step in genetic algorithm
    translation_step = {"rotation": 6, "translate": 1, "shear": 0.02, "zoom": 0.02,
                        "blur": 1, "brightness": 4, "contrast": 0.05}

    enable_optimize = True
    epoch_level_augment = False

    def __init__(self):
        pass

    def print_config(self):
        logger.info("=============== global config ===============")
        logger.info("queue length : " + str(self.queue_len))
        logger.info("mutate probability : " + str(self.prob_mutate))
        logger.info("the number of process (for multiprocessing) : " + str(self.num_processor))
        logger.info("coverage differential threshold : " + str(self.coverage_threshold))
        logger.info("enable transformation based on filter : " + str(self.enable_filters))
        logger.info("enable optimize : " + str(self.enable_optimize))

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


global_config = ExperimentalConfig()
