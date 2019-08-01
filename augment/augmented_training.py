"""
This program is designed to train a model based on augmented dataset
Author: Xiang Gao (xiang.gao@us.fujitsu.com)
Time: Sep, 21, 2018
"""

from dataset.gtsrb.train import GtsrbModel
from dataset.cifar10.train import Cifar10Model
from dataset.svhn.train import SVHN
from dataset.fashionmnist.train import FashionMnist
from dataset.imdb.train import IMDBModel
from dataset.utk.train import UTKModel
from dataset.kvasir.train import KvasirModel
import argparse
from config import ExperimentalConfig
from util import SAU, DATASET, logger


class AugmentedModel:

    def __init__(self, target=None):
        self.target = target

    def train(self, strategy=SAU.replace30, _model=None):

        x_train, y_train = self.target.load_original_data('train')
        x_val, y_val = self.target.load_original_data('val')

        # data_generator = DataGenerator(self.target, _model, x_train, y_train, 32, strategy)
        # data_generator = None
        model = self.target.train_dnn_model(_model=_model,
                                            x_train=x_train, y_train=y_train,
                                            x_val=x_val, y_val=y_val,
                                            train_strategy=strategy)
        del model

    def test(self, _model=None):
        x_test, y_test = self.target.load_original_test_data()
        model = self.target.load_model(_model[0], _model[1])
        self.target.test_dnn_model(model, "", self.target.preprocess_original_imgs(x_test), y_test)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Augmented Training.')
    parser.add_argument('-s', '--strategy', dest='strategy', type=str, nargs='+',
                        help='augmentation strategy, supported strategy:' + str(SAU.list()))
    parser.add_argument('-d', '--dataset', dest='dataset', type=str, nargs='+',
                        help='the name of dataset, support dataset:' + str(DATASET.list()))
    parser.add_argument('-q', '--queue', dest='queue', type=int, nargs='+', default=[10],
                        help='the length of queue for genetic algorithm (default 10)')
    parser.add_argument('-m', '--model', dest='model', type=int, nargs='+', default=[0],
                        help='selection of model')
    parser.add_argument('-t', '--start-point', dest='start_point', type=int, nargs='+', default=0,
                        help='the start point of epoch (default from epoch 0)')
    parser.add_argument('-r', '--threshold', dest='threshold', type=int, nargs='+', default=[3],
                        help='the loss threshold for selective augmentation (default 1e-3)')
    parser.add_argument('-e', '--epoch', dest='epoch', type=int, nargs='+', default=200,
                        help='the number of training epochs')
    parser.add_argument('-f', '--filter', action='store_true', dest='enable_filter',
                        help='enable filter transformation operators (zoom, blur, contrast, brightness)')
    parser.add_argument('-o', '--optimize', action='store_true', dest='enable_optimize',
                        help='enable optimize')

    args = parser.parse_args()

    if len(args.strategy) <= 0 or len(args.dataset) <= 0:
        logger.error(parser)
        exit(1)

    # augmentation strategy
    aug_strategy = args.strategy[0]
    if aug_strategy not in SAU.list():
        logger.error("unsupported strategy, please use --help to find supported ones")
        exit(1)
    # target dataset
    dataset = args.dataset[0]
    if dataset not in DATASET.list():
        logger.error("unsupported dataset, please use --help to find supported ones")
        exit(1)

    config = ExperimentalConfig.gen_config()

    config.queue_len = int(0.4*args.queue[0])
    if config.queue_len<2:
        config.queue_len = 2
    config.popsize = args.queue[0] - config.queue_len
    config.enable_filters = args.enable_filter
    config.enable_optimize = args.enable_optimize
    start_point = args.start_point[0]
    epoch = args.epoch[0]
    model_index = int(args.model[0])
    config.robust_threshold = 0.1**int(args.threshold[0])

    # initialize dataset
    dat = DATASET.get_name(dataset)
    if dat.value == DATASET.gtsrb.value:
        target0 = GtsrbModel('GTSRB', start_point, epoch)
    elif dat.value == DATASET.cifar10.value:
        target0 = Cifar10Model(start_point, epoch)
    elif dat.value == DATASET.fashionmnist.value:
        target0 = FashionMnist(start_point, epoch)
    elif dat.value == DATASET.svhn.value:
        target0 = SVHN("data", start_point, epoch)
        config.brightness_range=[0]
        config.contrast_range=[1]
    elif dat.value == DATASET.imdb.value:
        target0 = IMDBModel("dataset", start_point, epoch)
    elif dat.value == DATASET.utk.value:
        target0 = UTKModel("dataset", start_point, epoch)
    elif dat.value == DATASET.kvasir.value:
        target0 = KvasirModel("kvasir-dataset", start_point, epoch)
    else:
        raise Exception('unsupported dataset', dataset)

    ExperimentalConfig.save_config(config)
    config.print_config()

    atm = AugmentedModel(target0)

    logger.info("===========  " + aug_strategy + " on "
                + dataset + " dataset =========== ")

    _model_file = "models/" + dataset + aug_strategy + "_model"+str(model_index)+"_" + \
                  str(config.enable_filters) + "_O_" + str(config.enable_optimize) + \
                  "_model_" + str(model_index) + ".hdf5"
    _model0 = [model_index, _model_file]

    atm.train(SAU.get_name(aug_strategy), _model0)

