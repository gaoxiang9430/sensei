"""
This program extends Sequence, which is used to provide training data for model training process
Author: Xiang Gao (xiang.gao@us.fujitsu.com)
Time: Sep, 21, 2018
"""

from augmentor import *
import keras
import tensorflow as tf
import copy
from util import SAU, logger
from config import global_config as config
from ga_selector import GASelect
from neural_coverage import NeuralCoverage
import random
from operator import itemgetter
import time


class DataGenerator(keras.utils.Sequence):
    """Generates data for Keras"""
    def __init__(self, original_target=None, model=None, x_original_train=None, y_original_train=None,
                 batch_size=128, strategy=SAU.replace30, graph=None):

        self.au = Augmenter()
        self.batch_size = batch_size
        self.original_target = original_target
        self.x_original_train = x_original_train
        self.y_original_train = y_original_train
        self.strategy = strategy
        self.model = model
        self.graph = graph

        if config.enable_optimize:
            self.skipped_node = 0

        temp_x_original_train = copy.deepcopy(self.x_original_train)
        self.y_train = y_original_train
        if self.strategy.value == SAU.augment30.value:
            aug_x_train, self.y_train = self.au.random_augment(temp_x_original_train, self.y_train)
            self.x_train = self.original_target.preprocess_original_imgs(aug_x_train)
        else:
            self.x_train = original_target.preprocess_original_imgs(temp_x_original_train)

        if self.strategy.value == SAU.ga_loss.value:
            self.ga_selector = GASelect(self.x_original_train, self.y_original_train, self.original_target)
        elif self.strategy.value == SAU.ga_cov.value:
            self.nc = NeuralCoverage(model)
            self.ga_selector = GASelect(self.x_original_train, self.y_original_train)
        elif self.strategy.value == SAU.replace_worst_of_10_cov.value:
            self.nc = NeuralCoverage(model)
        elif self.strategy.value == SAU.replace_worst_of_10.value and config.enable_optimize:
            self.is_robust = [False] * len(x_original_train)

        self.index = 0
        self.total_time = 0
        self.predict_time = 0
        if config.enable_optimize:
            self.current_indices = range(len(x_original_train))

    def performance(self, model, md, x, y):
        pt = Perturbator()
        with self.graph.as_default():
            x_augs, y = self.au.worst_of_10(copy.deepcopy(x), y)
            y = y*10
            x_augs = md.preprocess_original_imgs(x_augs)
            start_time = time.time()
            y_predict = model.predict(x_augs)
            print("prediction time:", time.time() - start_time)

    def __len__(self):
        logger.info("size of batch: " + str(int(np.ceil(len(self.x_train) / self.batch_size))))
        """Denotes the number of batches per epoch"""
        return int(np.ceil(len(self.x_train) / self.batch_size))

    def cross_entropy(self, predictions, targets):
        ce = -np.sum(targets*np.log(predictions)+(1-targets)*np.log(1-predictions), axis=1)
        return ce

    def __getitem__(self, index):
        """Generate one batch of data according to the batch index"""
        start = index*self.batch_size
        end = (index+1)*self.batch_size
        if end > len(self.x_train):
            end = len(self.x_train)
        x = copy.deepcopy(self.x_train[start:end])
        y = self.y_train[start:end]

        if config.epoch_level_augment:  # or self.index % 2 == 0:
            return x, y

        if index % 50 == 0:
            self.original_target.test_dnn_model(self.model, "Before: Loss of new x: ", x, y)
        # perform batch level augmentation
        if self.strategy.value == SAU.replace_worst_of_10.value:
            x_origin = self.x_original_train[start:end]
            temp_x_original_train = copy.copy(x_origin)
            x_10, self.y_train = self.au.worst_of_10(temp_x_original_train, self.y_train)
            if config.enable_optimize:
                x, loss = self.optimized_select_worst(x_10, y, self.is_robust[start:end])
                max_loss = np.max(loss, axis=0)
                for j in range(len(max_loss)):
                    if max_loss[j] < 1e-4:  # this node becomes robust
                        self.is_robust[j+start] = True
            else:
                x, loss = self.select_worst(x_10, y)
            self.x_train[start:end] = x

        elif self.strategy.value == SAU.replace_worst_of_10_cov.value:
            temp_x_original_train = copy.copy(self.x_original_train[start:end])
            # randomly generate 10 candidates
            x_10, self.y_train = self.au.worst_of_10(temp_x_original_train, self.y_train)
            x = self.select_worst_cov(x, x_10, y)
            self.x_train[start:end] = x

        elif self.strategy.value == SAU.ga_loss.value:
            is_robust, x_n = self.ga_selector.generate_next_population(start, end)  # num_population * num_test

            if config.enable_optimize:
                x, loss = self.optimized_select_worst(x_n, y, is_robust)
            else:
                x, loss = self.select_worst(x_n, y)

            self.ga_selector.fitness(loss, start, end)
            self.x_train[start:end] = x

        elif self.strategy.value == SAU.ga_cov.value:
            x_n = self.ga_selector.generate_next_population(start, end)  # num_population * num_test
            x, loss = self.generate_cov(x, x_n, y)
            self.ga_selector.fitness(loss, start, end)
            self.x_train[start:end] = x

        if index % 50 == 0:
            self.original_target.test_dnn_model(self.model, "After: Loss of new x: ", x, y)
        return x, y

    def on_epoch_end(self):
        tf.reset_default_graph()

        if config.enable_optimize:
            print("number of skipped node is: ", self.skipped_node, len(self.y_train))
            self.is_robust = [False] * len(self.x_original_train)
        print("prediction time is: ", self.predict_time)
        print("selection time is: ", self.total_time)
        self.total_time = 0
        self.predict_time = 0
        self.index += 1

        # do not need to augment for the last epoch
        if self.index == self.original_target.epoch:
            return

        # left-right flip for Cifar-10
        if self.original_target.__class__.__name__ == "Cifar10Model":
            for i in range(len(self.x_original_train)):
                flip = random.choice([True, False])
                if flip:
                    self.x_original_train[i] = np.fliplr(self.x_original_train[i])
            logger.info("Flip done")

        # if config.epoch_level_augment and self.strategy.value == SAU.ga_cov.value:
        #     self.class_cov = self.nc.generate_cov_for_class(self.x_train, self.y_train)

        # if it is not epoch level augment, direct skip the following augmentation process

        """perturb the training sets after each epoch"""
        if self.strategy.value == SAU.original.value:
            # if config.data_set == "cifar10":
            if self.original_target.__class__.__name__ == "Cifar10Model":
                temp_x_original_train = copy.deepcopy(self.x_original_train)
                self.x_train = self.original_target.preprocess_original_imgs(temp_x_original_train)
            logger.info(" Training on original dataset!!!")

        elif self.strategy.value == SAU.replace30.value:
            temp_x_original_train = copy.deepcopy(self.x_original_train)
            temp_x_original_train, self.y_train = self.au.random_replace(temp_x_original_train, self.y_train)
            del self.x_train
            self.x_train = self.original_target.preprocess_original_imgs(temp_x_original_train)
            logger.info(" Augmentation replace30 Done!!!")

        elif self.strategy.value == SAU.augment30.value:
            temp_x_original_train = copy.deepcopy(self.x_original_train)
            temp_x_original_train, self.y_train = self.au.random_augment(temp_x_original_train,
                                                                         self.y_original_train)
            del self.x_train
            self.x_train = self.original_target.preprocess_original_imgs(temp_x_original_train)
            logger.info(" Augmentation augment30 Done!!!")

        elif self.strategy.value == SAU.replace40.value:
            del self.x_train
            temp_x_original_train = copy.deepcopy(self.x_original_train)
            self.x_train, self.y_train = self.au.random40_replace(temp_x_original_train, self.y_train)
            self.x_train = self.original_target.preprocess_original_imgs(self.x_train)
            logger.info(" Augmentation replace40 Done!!!")

        if not config.epoch_level_augment:
            return

        if self.strategy.value == SAU.replace_worst_of_10.value:
            if config.enable_optimize:
                logger.debug("the number of images to be augmented is : " + str(len(self.current_indices)))
                temp_x_original_train = copy.deepcopy(
                    list(itemgetter(*self.current_indices)(self.x_original_train)))
            else:
                temp_x_original_train = copy.deepcopy(self.x_original_train)
            # randomly generate 10 candidates
            x_10, self.y_train = self.au.worst_of_10(temp_x_original_train, self.y_train)
            logger.info("Generation Done!!!")
            # select the worst one based on loss
            if config.enable_optimize:
                self.select_worst_epoch(x_10, list(itemgetter(*self.current_indices)(self.y_train)))
            else:
                self.select_worst_epoch(x_10, self.y_train)
            logger.info(" Augmentation replace_worst_of_10 Done!!!")

        elif self.strategy.value == SAU.replace_worst_of_10_cov.value:
            temp_x_original_train = copy.deepcopy(self.x_original_train)
            # randomly generate 10 candidates
            x_10, self.y_train = self.au.worst_of_10(temp_x_original_train, self.y_train)
            logger.info("Generation Done!!!")
            self.select_worst_cov_epoch(x_10, self.y_train)
            logger.info(" Augmentation replace_worst_of_10_cov Done!!!")

        elif self.strategy.value == SAU.ga_loss.value:
            x_n = self.ga_selector.generate_next_population()  # num_population * num_test
            loss = self.generate_loss_epoch(x_n, self.y_train)
            self.ga_selector.fitness(loss)
            del loss
            del x_n[:]
            del x_n

        elif self.strategy.value == SAU.ga_cov.value:
            x_n = self.ga_selector.generate_next_population()  # num_population * num_test
            cov = self.generate_cov_epoch(x_n, self.y_train)
            self.ga_selector.fitness(cov)
            del cov
            del x_n[:]
            del x_n

        else:
            raise Exception('unsupported augment strategy', self.strategy)

    def select_worst_epoch(self, x_10=None, y=None):
        """Evaluate the loss of each image, and select the worst one based on loss"""
        loss = []
        for i in range(len(x_10)):
            x_10[i] = self.original_target.preprocess_original_imgs(x_10[i])
        with self.graph.as_default():
            for i in range(10):
                logger.debug("processing round " + str(i))
                set_i = np.array(x_10)[:, i]

                y_predict_temp = self.model.predict(set_i)
                y_true1 = np.array(y, dtype='float32')

                y_true1 = tf.convert_to_tensor(y_true1)
                y_pred1 = tf.convert_to_tensor(y_predict_temp)

                loss1 = keras.losses.categorical_crossentropy(y_true1, y_pred1)
                loss1 = keras.backend.get_value(loss1)
                loss.append(loss1)
            logger.debug("loss generation done!!!")
            y_argmax = np.argmax(loss, axis=0)
            y_max = np.max(loss, axis=0)

        logger.debug("length of y_argmax: " + str(len(y_argmax)))
        if config.enable_optimize:
            temp_indices = copy.deepcopy(self.current_indices)
        for j in range(len(y_argmax)):
            if config.enable_optimize:
                x_index = temp_indices[j]
                index = y_argmax[j]
                self.x_train[x_index] = x_10[j][index]  # update x_train (not good design)
                if y_max[j] < 1e-3:
                    self.current_indices.remove(x_index)
            else:
                index = y_argmax[j]
                self.x_train[j] = x_10[j][index]  # update x_train (not good design)
        return self.x_train

    def select_worst(self, x_10=None, y=None):
        """Evaluate the loss of each image, and select the worst one based on loss"""
        for i in range(len(x_10)):
            x_10[i] = self.original_target.preprocess_original_imgs(x_10[i])
        n = len(x_10[0])
        num_perturb = len(x_10)

        with self.graph.as_default():
            # set_i = np.array(x_10)[:, i]
            x_10_temp = copy.deepcopy(x_10)
            shape = (n * num_perturb,) + self.original_target.input_shape
            s_time = time.time()
            y_predict_temp = self.model.predict(np.asarray(x_10_temp).reshape(shape))
            self.predict_time += time.time()-s_time
            del x_10_temp
            ss_start = time.time()
            y_true1 = list(y) * num_perturb
            '''
            y_true1 = np.array(y_true1, dtype='float32')
            y_true1 = tf.convert_to_tensor(y_true1)
            y_pred1 = tf.convert_to_tensor(y_predict_temp)
            loss1 = keras.losses.categorical_crossentropy(y_true1, y_pred1)
            loss = keras.backend.get_value(loss1)
            '''
            y_pred1 = np.clip(np.array(y_predict_temp, dtype='float64'), 1e-8, 1-1e-8)
            y_true1 = np.array(y_true1, dtype='float64')
            loss = self.cross_entropy(y_pred1, y_true1)

            loss_all = np.asarray(loss).reshape(num_perturb, n)
            max_loss = loss_all[0]
            x_origin = x_10[0]

            for i in range(1, num_perturb):
                loss = loss_all[i]
                set_i = x_10[i]

                idx = (loss > max_loss)
                max_loss = np.maximum(loss, max_loss)
                idx = np.expand_dims(idx, axis=-1)
                idx = np.expand_dims(idx, axis=-1)
                idx = np.expand_dims(idx, axis=-1)  # shape (bsize, 1, 1, 1)
                x_origin = np.where(idx, set_i, x_origin, )  # shape (bsize, 32, 32, 3)
            self.total_time += time.time()-ss_start
        return x_origin, loss_all

    def optimized_select_worst(self, x_10=None, y=None, is_robust=None):
        """Evaluate the loss of each image, and select the worst one based on loss"""
        for i in range(len(x_10)):
            x_10[i] = self.original_target.preprocess_original_imgs(x_10[i])
        x_10 = np.asarray(x_10)
        n = len(x_10[0])
        logger.debug("original shape", np.shape(x_10))

        robust_x = []
        robust_y = []
        unrobust_x = []
        unrobust_y = []

        for i in range(n):
            if is_robust[i]:
                robust_x.append(x_10[0][i])
                robust_y.append(y[i])
            else:
                unrobust_x.append(x_10[:, i])
                unrobust_y.append(y[i])

        self.skipped_node += len(robust_y)

        if len(unrobust_x) != 0:
            unrobust_x = np.swapaxes(unrobust_x, 0, 1)
        logger.debug("length of robust", len(robust_y))
        logger.debug("length of unrobust", np.shape(np.asarray(unrobust_x)))

        num_perturb = len(unrobust_x)
        if num_perturb == 0:
            n = 0
        else:
            n = len(unrobust_x[0])

        with self.graph.as_default():
            if len(unrobust_x) != 0:
                x_10_temp = copy.deepcopy(unrobust_x)
                shape = (n * num_perturb,) + self.original_target.input_shape
                training_data = np.asarray(x_10_temp).reshape(shape)
                if len(robust_x) != 0:
                    training_data = np.concatenate((training_data, robust_x), axis=0)
            else:
                training_data = np.asarray(robust_x)
            logger.debug("length of training_data", np.shape(training_data))
            start_time = time.time()
            y_predict_temp = self.model.predict(training_data)
            self.predict_time += time.time()-start_time

            y_true1 = list(unrobust_y) * num_perturb + robust_y
            #y_true1 = np.array(y_true1, dtype='float32')
            #y_true1 = tf.convert_to_tensor(y_true1)
            #y_pred1 = tf.convert_to_tensor(y_predict_temp)
            #loss1 = keras.losses.categorical_crossentropy(y_true1, y_pred1)
            #loss = keras.backend.get_value(loss1)

            y_pred1 = np.clip(np.array(y_predict_temp, dtype='float64'), 1e-8, 1-1e-8)
            y_true1 = np.array(y_true1, dtype='float64')
            loss = self.cross_entropy(y_pred1, y_true1)

            boundary = -len(robust_y)
            robust_loss = []
            if boundary != 0:
                robust_loss = loss[boundary:]
                unrobust_loss = loss[:boundary]
            else:
                unrobust_loss = loss
            unrobust_loss = np.asarray(unrobust_loss).reshape(num_perturb, n)

            loss_all = np.zeros((len(x_10), len(x_10[0])))
            robust_index = 0
            unrobust_index = 0
            for i in range(len(x_10[0])):
                if is_robust[i]:
                    loss_all[:, i] = robust_loss[robust_index:robust_index+1]*len(x_10)
                    robust_index += 1
                else:
                    loss_all[:, i] = unrobust_loss[:, unrobust_index]
                    unrobust_index += 1

            logger.debug("loss shape", np.shape(loss_all))

            max_loss = loss_all[0]
            x_origin = x_10[0]

            for i in range(1, num_perturb):
                loss = loss_all[i]
                set_i = x_10[i]

                idx = (loss > max_loss)
                max_loss = np.maximum(loss, max_loss)
                idx = np.expand_dims(idx, axis=-1)
                idx = np.expand_dims(idx, axis=-1)
                idx = np.expand_dims(idx, axis=-1)  # shape (bsize, 1, 1, 1)
                x_origin = np.where(idx, set_i, x_origin, )  # shape (bsize, 32, 32, 3)

        return x_origin, loss_all

    def select_worst_cov_epoch(self, x_10=None, y=None):
        """select worst image based on neural coverage"""
        cov_diff = []
        for i in range(len(x_10)):
            x_10[i] = self.original_target.preprocess_original_imgs(x_10[i])
        with self.graph.as_default():
            origin_cov = self.nc.get_layer_output(self.x_train)
            # class_cov = self.nc.generate_cov_for_class(origin_cov, y)
            # del origin_cov[:]
            # del origin_cov
            for i in range(10):
                logger.debug("processing round " + str(i))
                set_i = np.array(x_10)[:, i]
                cov = self.nc.get_layer_output(set_i)
                cov_diff_i = self.nc.compare_output(cov, origin_cov, y)
                # cov_diff_i = self.nc.compare_class_output(cov, class_cov, y)
                del cov[:]
                del cov
                cov_diff.append(cov_diff_i)
            logger.debug("coverage generation done!!!")
        y_argmax = np.argmax(cov_diff, axis=0)
        logger.debug("length of y_argmax: " + str(len(y_argmax)))
        print(y_argmax)
        for j in range(len(y_argmax)):
            index = y_argmax[j]
            self.x_train[j] = x_10[j][index]  # update x_train (not good design)

    def select_worst_cov(self, origin_x=None, x_10=None, y=None):

        """Evaluate the loss of each image, and select the worst one based on loss"""
        for i in range(len(x_10)):
            x_10[i] = self.original_target.preprocess_original_imgs(x_10[i])
        n = len(x_10)
        num_perturb = len(x_10[0])

        with self.graph.as_default():
            x_10_temp = copy.copy(x_10)
            # calculate the coverage for all inputs together (to improve performance)
            shape = (n * num_perturb,) + self.original_target.input_shape
            # add original image to the candidate set
            # add original image as baseline
            x_pre = np.concatenate((origin_x, origin_x, np.asarray(x_10_temp).reshape(shape)), axis=0)
            y_pre = np.concatenate((y, np.repeat(y, num_perturb, axis=0)), axis=0)
            cov_diff = self.nc.compare_output2(x_pre, y_pre, n)

            cov_diff_all = np.asarray(cov_diff[n:]).reshape(n, num_perturb)
            max_diff = cov_diff[0:n]
            x_origin = origin_x
            # max_diff = cov_diff_all[:, 0]
            x_10 = np.asarray(x_10)
            # x_origin = x_10[:, 0]
            for i in range(0, num_perturb):
                # diff = np.array(self.nc.compare_output(new_cov[:, i], origin_cov, y))
                diff = cov_diff_all[:, i]
                set_i = x_10[:, i]

                idx = (diff > max_diff)
                max_diff = np.maximum(diff, max_diff)
                idx = np.expand_dims(idx, axis=-1)
                idx = np.expand_dims(idx, axis=-1)
                idx = np.expand_dims(idx, axis=-1)  # shape (bsize, 1, 1, 1)
                x_origin = np.where(idx, set_i, x_origin, )  # shape (bsize, 32, 32, 3)
        return x_origin

    def select_worst_cov2(self, origin_x=None, x_10=None, y=None):

        """Evaluate the loss of each image, and select the worst one based on loss"""
        for i in range(len(x_10)):
            x_10[i] = self.original_target.preprocess_original_imgs(x_10[i])
        n = len(x_10)
        num_perturb = len(x_10[0])

        with self.graph.as_default():
            x_10_temp = copy.copy(x_10)
            # calculate the coverage for all inputs together (for performance)
            shape = (n * num_perturb,) + self.original_target.input_shape
            # add original image to the candidate set
            # add original image as baseline
            x_pre = np.concatenate((origin_x, origin_x, np.asarray(x_10_temp).reshape(shape)), axis=0)

            cov_diff = self.nc.compare_output2(x_pre, np.repeat(y, num_perturb+1, axis=0), n)
            # max_diff = cov_diff[0:n]
            # x_origin = origin_x
            cov_diff_all = np.asarray(cov_diff[n:]).reshape(n, num_perturb)
            x_10 = np.asarray(x_10)
            x_origin = x_10[:, 0]
            max_diff = cov_diff_all[:, 0]
            for i in range(1, num_perturb):
                # diff = np.array(self.nc.compare_output(new_cov[:, i], origin_cov, y))
                diff = cov_diff_all[:, i]
                set_i = x_10[:, i]

                idx = (diff > max_diff)
                max_diff = np.maximum(diff, max_diff)
                idx = np.expand_dims(idx, axis=-1)
                idx = np.expand_dims(idx, axis=-1)
                idx = np.expand_dims(idx, axis=-1)  # shape (bsize, 1, 1, 1)
                x_origin = np.where(idx, set_i, x_origin, )  # shape (bsize, 32, 32, 3)
        return x_origin

    def generate_loss_epoch(self, x_n=None, y=None):
        """generate loss for each population"""
        loss = []
        for i in range(len(x_n)):
            x_n[i] = self.original_target.preprocess_original_imgs(x_n[i])
        with self.graph.as_default():
            for i in range(len(x_n)):

                y_predict_temp = self.model.predict(x_n[i])
                y_true1 = np.array(y, dtype='float32')

                y_true1 = tf.convert_to_tensor(y_true1)
                y_pred1 = tf.convert_to_tensor(y_predict_temp)

                loss1 = keras.losses.categorical_crossentropy(y_true1, y_pred1)
                loss1 = keras.backend.get_value(loss1)
                loss.append(loss1)     # num_population * num_test
            y_argmax = np.argmax(loss, axis=0)

        logger.debug("length of y_argmax: " + str(str(len(y_argmax))))
        print(y_argmax)
        for j in range(len(y_argmax)):
            index = y_argmax[j]
            self.x_train[j] = x_n[index][j]   # update x_train (not good design)
        return loss

    def generate_cov_epoch(self, x_n=None, y=None):
        """generate loss for each population"""
        cov_diff = []
        for i in range(len(x_n)):
            x_n[i] = self.original_target.preprocess_original_imgs(x_n[i])
        with self.graph.as_default():
            origin_cov = self.nc.get_layer_output(self.x_train)
            class_cov = self.nc.generate_cov_for_class(origin_cov, y)
            del origin_cov[:]
            del origin_cov
            for i in range(len(x_n)):
                logger.debug("processing round " + str(i))
                cov = self.nc.get_layer_output(x_n[i])
                # cov_diff_i = self.nc.compare_output(cov, origin_cov, y)
                cov_diff_i = self.nc.compare_class_output(cov, class_cov, y)
                del cov[:]
                del cov
                cov_diff.append(cov_diff_i)
            y_argmax = np.argmax(cov_diff, axis=0)

        logger.debug("length of y_argmax: " + str(str(len(y_argmax))))
        print(y_argmax)
        for j in range(len(y_argmax)):
            index = y_argmax[j]
            self.x_train[j] = x_n[index][j]   # update x_train (not good design)
        return cov_diff

    def generate_cov(self, origin_x=None, x_n=None, y=None):
        """generate loss for each population"""
        for i in range(len(x_n)):
            x_n[i] = self.original_target.preprocess_original_imgs(x_n[i])
        n = len(x_n[0])
        num_perturb = len(x_n)
        with self.graph.as_default():
            x_10_temp = copy.copy(x_n)
            shape = (n*num_perturb,)+self.original_target.input_shape
            x_pre = np.concatenate((origin_x, np.asarray(x_10_temp).reshape(shape)), axis=0)
            y_pre = list(y) * num_perturb
            y_pre = np.array(y_pre, dtype='float32')
            cov_diff = self.nc.compare_output2(x_pre, y_pre, n)

            cov_diff_all = np.asarray(cov_diff).reshape(num_perturb, n)
            x_10 = np.asarray(x_n)
            x_origin = x_10[0]
            max_diff = cov_diff_all[0]

            for i in range(1, num_perturb):
                diff = cov_diff_all[1]
                set_i = x_10[i]

                idx = (diff > max_diff)
                max_diff = np.maximum(diff, max_diff)
                idx = np.expand_dims(idx, axis=-1)
                idx = np.expand_dims(idx, axis=-1)
                idx = np.expand_dims(idx, axis=-1)  # shape (bsize, 1, 1, 1)
                x_origin = np.where(idx, set_i, x_origin, )  # shape (bsize, 32, 32, 3)

        return x_origin, cov_diff_all

