"""
This program is designed to train a model based on augmented dataset
Author: Xiang Gao (xiang.gao@us.fujitsu.com)
Time: Sep, 21, 2018
"""

from dataset.gtsrb.train import GtsrbModel
from dataset.cifar10.train import Cifar10Model
import argparse
from config import global_config as config
from util import SAU, DATASET, logger


class AugmentedModel:

    def __init__(self, target=None):
        self.target = target

    def train(self, strategy=SAU.replace30, _model=None):

        x_train, y_train = self.target.load_original_data('train')
        x_val, y_val = self.target.load_original_data('val')

        # data_generator = DataGenerator(self.target, _model, x_train, y_train, 32, strategy)
        # data_generator = None
        self.target.train_dnn_model(_model=_model,
                                    x_train=x_train, y_train=y_train,
                                    x_val=x_val, y_val=y_val,
                                    train_strategy=strategy)

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
    parser.add_argument('-q', '--queue', dest='queue', type=int, nargs='+', default=10,
                        help='the length of queue for genetic algorithm (default 10)')
    parser.add_argument('-t', '--start-point', dest='start_point', type=int, nargs='+', default=0,
                        help='the start point of epoch (default from epoch 0)')
    parser.add_argument('-e', '--epoch', dest='epoch', type=int, nargs='+', default=200,
                        help='the number of training epochs')
    parser.add_argument('-f', '--filter', action='store_true', dest='enable_filter',
                        help='enable filter transformation operators (zoom, blur, contrast, brightness)')

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

    config.queue_len = args.queue
    config.enable_filters = args.enable_filter
    start_point = args.start_point[0]
    epoch = args.epoch[0]

    # initialize dataset
    dat = DATASET.get_name(dataset)
    if dat.value == DATASET.gtsrb.value:
        target0 = GtsrbModel('GTSRB', start_point, epoch)
    elif dat.value == DATASET.cifar10.value:
        target0 = Cifar10Model(start_point, epoch)
    else:
        raise Exception('unsupported dataset', dataset)

    atm = AugmentedModel(target0)

    logger.info("===========  " + aug_strategy + " on "
                + dataset + " dataset =========== ")
    _model_file = "models/" + dataset + aug_strategy + "_model_" + \
                  str(config.enable_filters) + ".hdf5"
    _model0 = [0, _model_file]

    config.print_config()

    atm.train(SAU.get_name(aug_strategy), _model0)

    # _model0 = [0, "models/gtsrb.oxford.model.hdf5"]
    # atm.train(SAU.original, _model30)
    # atm.test(_model0)

    # _model30 = [0, "models/gtsrb.oxford.replace30_model.hdf5"]
    # atm.train(SAU.replace30, _model30)
    # atm.test(_model30)

    # _model40 = [0, "models/gtsrb.oxford.replace40_model.hdf5"]
    # atm.train(SAU.replace40, _model40)
    # atm.test(_model40)

    # _model_w_10 = [0, "models/gtsrb.oxford.w_10_model.hdf5"]
    # atm.train(SAU.replace_worst_of_10, _model_w_10)
    # atm.test(_model_w_10)
