"""
This program is designed to attach a given models
Author: Xiang Gao (xiang.gao@us.fujitsu.com)
Time: Sep, 21, 2018
"""

from dataset.gtsrb.train import GtsrbModel
from dataset.cifar10.train import Cifar10Model
from dataset.fashionmnist.train import FashionMnist
from dataset.svhn.train import SVHN
from dataset.imdb.train import IMDBModel
from dataset.utk.train import UTKModel
from dataset.kvasir.train import KvasirModel
from perturbator import Perturbator
from config import ExperimentalConfig
from util import SAT, SAU, DATASET, logger
import numpy as np
import copy
import argparse


class AttackModel:

    def __init__(self, target=None):
        self.target = target
        self.config = ExperimentalConfig.gen_config()

    @staticmethod
    def predict(model, x, y):
        y_predict = model.predict(x)
        return np.argmax(y_predict, axis=1) == np.argmax(y, axis=1)

    def attack(self, strategy=SAT.random, _model=None, print_label=""):
        x_test, y_test = self.target.load_original_test_data()
        # x_test, y_test = self.target.load_original_data("train")
        if self.config.enable_filters:
            part_len = len(x_test)/4
            x_test = x_test[0:part_len]
            y_test = y_test[0:part_len]
            
        model = self.target.load_model(_model[0], _model[1])

        pt = Perturbator()
        if strategy.value == SAT.original.value:
            self.target.test_dnn_model(model, print_label,
                                       self.target.preprocess_original_imgs(x_test), y_test)
        elif strategy.value == SAT.fix.value:
            x_test, y_test = pt.fix_perturb(x_test, y_test, -30, -3, -0.2, 0.9, 0, -32, 0.8)
            self.target.test_dnn_model(model, print_label,
                                       self.target.preprocess_original_imgs(x_test), y_test)
        elif strategy.value == SAT.random.value:
            x_test, y_test = pt.random_perturb(x_test, y_test)
            self.target.test_dnn_model(model, print_label,
                                       self.target.preprocess_original_imgs(x_test), y_test)
        elif strategy.value == SAT.random150.value:
            total_loss = 0
            total_acc = 0
            for p1 in range(50):
                temp_x_test = copy.deepcopy(x_test)
                x_perturbed_test, y_test = pt.random_perturb(temp_x_test, y_test)
                loss, acc = self.target.test_dnn_model(model, "Random " + str(p1) + " " + print_label,
                                                       self.target.preprocess_original_imgs(x_perturbed_test),
                                                       y_test)
                total_acc += acc
                total_loss += loss
            # calculate the average value of the grid perturbations
            print(print_label, ' - Test loss:', total_loss/50)
            print(print_label, ' - Test accuracy:', total_acc/50)

        elif strategy.value == SAT.rotate.value:
            correct = 0
            global_acc = 0
            for i in range(len(x_test)):
                grid_x_i = []
                grid_y_i = []
                for p1 in self.config.rotation_range:
                    x_i = copy.deepcopy(x_test[i])
                    x_i = pt.fix_perturb_img(x_i, p1)
                    grid_x_i.append(x_i)
                    grid_y_i.append(y_test[i])
                # current_perturb = print_label + "_i"
                current_perturb = "mute"
                loss, acc = self.target.test_dnn_model(model, current_perturb,
                                                       self.target.preprocess_original_imgs(grid_x_i),
                                                       np.array(grid_y_i))
                del grid_x_i
                del grid_y_i
                global_acc += acc
                if acc == 1:
                    correct += 1
                if i % 1000 == 0:
                    logger.info(print_label + ' - Test accuracy: ' + str(correct / float(i+1)))
                    logger.info(print_label + ' - Global accuracy: ' + str(global_acc / float(i+1)))
            logger.info(print_label + ' - Test accuracy: ' + str(correct / float(len(x_test))))
            logger.info(print_label + ' - Global accuracy: ' + str(global_acc / float(len(x_test))))

        elif strategy.value == SAT.translate.value:
            correct = 0
            global_acc = 0
            for i in range(len(x_test)):
                grid_x_i = []
                grid_y_i = []
                for p2 in self.config.translate_range:
                    for p2_v in self.config.translate_range:
                        x_i = copy.deepcopy(x_test[i])
                        x_i = pt.fix_perturb_img(x_i, 0, p2, p2_v)
                        grid_x_i.append(x_i)
                        grid_y_i.append(y_test[i])
                # current_perturb = print_label + "_i"
                current_perturb = "mute"
                loss, acc = self.target.test_dnn_model(model, current_perturb,
                                                       self.target.preprocess_original_imgs(grid_x_i),
                                                       np.array(grid_y_i))
                del grid_x_i
                del grid_y_i
                global_acc += acc
                if acc == 1:
                    correct += 1
                if i % 1000 == 0:
                    logger.info(print_label + ' - Test accuracy: ' + str(correct / float(i+1)))
                    logger.info(print_label + ' - Global accuracy: ' + str(global_acc / float(i+1)))
            logger.info(print_label + ' - Test accuracy: ' + str(correct / float(len(x_test))))
            logger.info(print_label + ' - Global accuracy: ' + str(global_acc / float(len(x_test))))

        elif strategy.value == SAT.shear.value:
            correct = 0
            global_acc = 0
            for i in range(len(x_test)):
                grid_x_i = []
                grid_y_i = []
                for p3 in self.config.shear_range:
                    x_i = copy.deepcopy(x_test[i])
                    x_i = pt.fix_perturb_img(x_i, 0, 0, 0, p3)
                    grid_x_i.append(x_i)
                    grid_y_i.append(y_test[i])
                # current_perturb = print_label + "_i"
                current_perturb = "mute"
                loss, acc = self.target.test_dnn_model(model, current_perturb,
                                                       self.target.preprocess_original_imgs(grid_x_i),
                                                       np.array(grid_y_i))
                del grid_x_i
                del grid_y_i
                global_acc += acc
                if acc == 1:
                    correct += 1
                if i % 1000 == 0:
                    logger.info(print_label + ' - Test accuracy: ' + str(correct / float(i+1)))
                    logger.info(print_label + ' - Global accuracy: ' + str(global_acc / float(i+1)))
            logger.info(print_label + ' - Test accuracy: ' + str(correct / float(len(x_test))))
            logger.info(print_label + ' - Global accuracy: ' + str(global_acc / float(len(x_test))))

        elif strategy.value == SAT.grid1.value:
            correct = 0
            for i in range(len(x_test)):
                is_correct = True
                for p1 in self.config.rotation_range[::30]:
                    for p2 in self.config.translate_range[::3]:
                        for p2_v in self.config.translate_range[::3]:
                            for p3 in self.config.shear_range[::20]:
                                if self.config.enable_filters:
                                    for p4 in self.config.zoom_range[::10]:
                                        for p5 in self.config.blur_range[::1]:
                                            for p6 in self.config.brightness_range[::16]:
                                                for p7 in self.config.contrast_range[::8]:
                                                    x_i = copy.deepcopy(x_test[i])
                                                    x_i = pt.fix_perturb_img(x_i, p1, p2, p2_v, p3, p4, p5, p6, p7)
                                                    loss, acc = self.target.test_dnn_model(
                                                                model, "mute",
                                                                self.target.preprocess_original_imgs([x_i]),
                                                                np.array([y_test[i]]))
                                                    if acc != 1:
                                                        print(p1, p2, p2_v, p3, p4, p5, p6, p7)
                                                        is_correct = False
                                else:
                                    x_i = copy.deepcopy(x_test[i])
                                    x_i = pt.fix_perturb_img(x_i, p1, p2, p2_v, p3)
                                    loss, acc = self.target.test_dnn_model(model, "mute",
                                                                           self.target.preprocess_original_imgs([x_i]),
                                                                           np.array([y_test[i]]))
                                    if acc != 1:
                                        print(p1, p2, p2_v, p3)
                                        is_correct = False
                if not is_correct:
                    correct += 1
                print("============= "+str(i)+" ===============")
            print("accuracy is " + str(float(correct)/len(x_test)))

        elif strategy.value == SAT.grid2.value:
            correct = 0
            incorrect = 0
            incorrect_node = 0
            global_acc = 0
            for i in range(len(x_test)):
                grid_x_i = []
                grid_y_i = []
                total_pert = 0
                for p1 in self.config.rotation_range[::30]:
                    for p2 in self.config.translate_range[::3]:
                        for p2_v in self.config.translate_range[::3]:
                            for p3 in self.config.shear_range[::20]:
                                if self.config.enable_filters:
                                    for p4 in self.config.zoom_range[::10]:
                                        for p5 in self.config.blur_range[::1]:
                                            for p6 in self.config.brightness_range[::16]:
                                                for p7 in self.config.contrast_range[::8]:
                                                    x_i = copy.deepcopy(x_test[i])
                                                    x_i = pt.fix_perturb_img(x_i, p1, p2, p2_v, p3, p4, p5, p6, p7)
                                                    grid_x_i.append(x_i)
                                                    grid_y_i.append(y_test[i])
                                                    total_pert += 1
                                else:
                                    x_i = copy.deepcopy(x_test[i])
                                    x_i = pt.fix_perturb_img(x_i, p1, p2, p2_v, p3)
                                    grid_x_i.append(x_i)
                                    grid_y_i.append(y_test[i])
                                    total_pert += 1
                # current_perturb = print_label + "_i"
                current_perturb = "mute"
                loss, acc = self.target.test_dnn_model(model, current_perturb,
                                                       self.target.preprocess_original_imgs(grid_x_i),
                                                       np.array(grid_y_i))
                del grid_x_i
                del grid_y_i
                global_acc += acc
                if acc == 1:
                    correct += 1
                else:
                    incorrect += 1
                    incorrect_node += (1-acc)*total_pert
                '''
                if i % 1000 == 0:
                    logger.info(print_label + ' - Test accuracy: ' + str(correct / float(i+1)))
                    if incorrect != 0:
                        logger.info(print_label+' - average number of failed perturbation for each misclassified image:'
                                    + str(incorrect_node / float(i+1)))
                    logger.info(print_label + ' - Global accuracy: ' + str(global_acc / float(i+1)))
                '''
            logger.info(print_label + ' - robust accuracy: ' + str(correct / float(len(x_test))))
            logger.info(print_label + ' - Global accuracy: ' + str(global_acc / float(len(x_test))))
            if incorrect != 0:
                logger.info(print_label + ' - average number of failed perturbation for each misclassified image: '
                            + str(incorrect_node / float(i+1)))

        else:
            raise Exception("Unsupported attack strategy")
        del model


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Augmented Training.')
    parser.add_argument('strategy', type=str,
                        help='augmentation strategy, supported strategy:' + str(SAU.list()))
    parser.add_argument('dataset',  type=str,
                        help='the name of dataset, support dataset:' + str(DATASET.list()))
    parser.add_argument('-f', '--filter', action='store_true', dest='enable_filter',
                        help='enable filter transformation operators (zoom, blur, contrast, brightness)')
    parser.add_argument('-o', '--optimize', action='store_true', dest='enable_optimize',
                        help='enable optimize')
    parser.add_argument('-m', '--model', dest='model', type=int, default=0,
                        help='selection of model')

    args = parser.parse_args()
    if len(args.strategy) <= 0 or len(args.dataset) <= 0:
        logger.error(parser)
        exit(1)

    # augmentation strategy
    aug_strategy = args.strategy
    if aug_strategy not in SAU.list():
        print("unsupported strategy, please use --help to find supported ones")
        exit(1)
    # target dataset
    dataset = args.dataset
    if dataset not in DATASET.list():
        print("unsupported dataset, please use --help to find supported ones")
        exit(1)

    config = ExperimentalConfig.gen_config()
    config.enable_filters = args.enable_filter
    config.enable_optimize = args.enable_optimize

    model_index = int(args.model)

    # initialize dataset
    dat = DATASET.get_name(dataset)
    if dat.value == DATASET.gtsrb.value:
        target0 = GtsrbModel(source_dir='GTSRB')
    elif dat.value == DATASET.cifar10.value:
        target0 = Cifar10Model()
    elif dat.value == DATASET.fashionmnist.value:
        target0 = FashionMnist()
    elif dat.value == DATASET.svhn.value:
        target0 = SVHN()
        config.brightness_range=[0]
        config.contrast_range=[1]
    elif dat.value == DATASET.imdb.value:
        target0 = IMDBModel("dataset")
    elif dat.value == DATASET.utk.value:
        target0 = UTKModel("dataset")
    elif dat.value == DATASET.kvasir.value:
        target0 = KvasirModel()
    
    else:
        raise Exception('unsupported dataset', dataset)

    ExperimentalConfig.save_config(config)
    config.print_config()

    logger.info("=========== attack " + aug_strategy + " of "
                + dataset + " dataset ===========")

    am = AttackModel(target=target0)

    _model_file = "models/" + dataset + aug_strategy + "_model"+str(model_index)+"_" + \
                  str(config.enable_filters) + "_O_" + str(config.enable_optimize) + \
                  "_model_" + str(model_index) + ".hdf5"

    _model0 = [model_index, _model_file]

    print("===================== test " + aug_strategy + " model =====================")
    print("\n===================== traditional accuracy =====================")
    am.attack(SAT.original, _model0, aug_strategy+" model")
    print("\n===================== random attack =====================")
    am.attack(SAT.random, _model0, aug_strategy + " model")
    '''
    print("\n===================== rotate =====================")
    am.attack(SAT.rotate, _model0, "rotate attack " + aug_strategy + " model")

    print("\n===================== translate =====================")
    am.attack(SAT.translate, _model0, "translate attack " + aug_strategy + " model")

    print("\n===================== shear =====================")
    am.attack(SAT.shear, _model0, "shear attack " + aug_strategy + " model")
    '''
    print("\n===================== grid attack =====================")
    am.attack(SAT.grid2, _model0, aug_strategy + " model")
