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
from config import ExperimentalConfig
from ga_selector import GASelect
from neural_coverage import NeuralCoverage
import random
from operator import itemgetter
import time


class DataGenerator(keras.utils.Sequence):
    """Generates data for Keras"""
    def __init__(self, original_target=None, model=None, x_original_train=None, y_original_train=None,
                 batch_size=128, strategy=SAU.replace30, graph=None):

        self.config = ExperimentalConfig.gen_config()
        self.au = Augmenter()
        self.batch_size = batch_size
        self.original_target = original_target
        self.x_original_train = x_original_train
        self.y_original_train = y_original_train
        self.strategy = strategy
        self.model = model
        self.graph = graph

        if self.config.enable_optimize:
            self.skipped_node = 0

            self.is_robust = [False] * len(x_original_train)
            self.prev_is_robust = [False] * len(x_original_train)

        # statistic record
        self.total_time = 0
        self.predict_time = 0
        self.label_record = [[0 for x in range(original_target.num_classes)]
                             for y in range(original_target.num_classes)]

        temp_x_original_train = copy.deepcopy(self.x_original_train)
        self.y_train = y_original_train
        self.x_train = original_target.preprocess_original_imgs(temp_x_original_train)

        if self.strategy.value == SAU.ga_loss.value:
            self.ga_selector = GASelect(self.x_original_train, self.y_original_train, self.original_target)
        elif self.strategy.value == SAU.ga_cov.value:
            self.nc = NeuralCoverage(model)
            self.ga_selector = GASelect(self.x_original_train, self.y_original_train, self.original_target)
 

    def cross_entropy(self, predictions, targets):
        ce = -np.sum(targets*np.log(predictions)+(1-targets)*np.log(1-predictions), axis=1)
        return ce

    def __len__(self):
        logger.info("size of batch: " + str(int(np.ceil(len(self.x_train) / self.batch_size))))
        """Denotes the number of batches per epoch"""
        return int(np.ceil(len(self.x_train) / self.batch_size))

    def __getitem__(self, index):
        """Generate one batch of data according to the batch index"""
        start = index*self.batch_size
        end = (index+1)*self.batch_size
        if end > len(self.x_train):
            end = len(self.x_train)

        x = copy.deepcopy(self.x_train[start:end])
        y = self.y_train[start:end]

        # regularly show debug information
        if index % 50 == 0:
            self.original_target.test_dnn_model(self.model, "Before: Loss of new x: ", x, y)

        # perform batch level augmentation
        if self.strategy.value == SAU.replace_worst_of_10.value:
            x_origin = self.x_original_train[start:end]
            temp_x_original_train = copy.copy(x_origin)
            # generate 10 perturbations
            x_10, self.y_train = self.au.worst_of_10(temp_x_original_train, self.y_train)
            if self.config.enable_optimize:
                x, loss = self.optimized_select_worst(x_10, y, self.prev_is_robust[start:end])
                max_loss = np.max(loss, axis=0)
                for j in range(len(max_loss)):
                    if max_loss[j] < self.config.robust_threshold:  # this node is point-wise robust
                        self.is_robust[j+start] = True
            else:
                x, loss = self.select_worst(x_10, y)
            self.x_train[start:end] = x

        elif self.strategy.value == SAU.ga_loss.value:
            # generate perturbations using GA
            is_robust, x_n = self.ga_selector.generate_next_population(start, end)  # num_population * num_test
            if self.config.robust_basedon_acc:
                is_robust = self.prev_is_robust[start:end]
            if self.config.enable_optimize:
                x, loss, predict_true = self.optimized_select_worst(x_n, y, is_robust)
                if self.config.robust_basedon_acc:
                    for j in range(len(predict_true)):
                        self.is_robust[j+start] = predict_true[j]
            else:
                x, loss = self.select_worst(x_n, y)

            self.ga_selector.fitness(loss, start, end)
            self.x_train[start:end] = x

        elif self.strategy.value == SAU.ga_cov.value:
            is_robust, x_n = self.ga_selector.generate_next_population(start, end)  # num_population * num_test
            x, loss = self.generate_cov(x, x_n, y)
            self.ga_selector.fitness(loss, start, end)
            self.x_train[start:end] = x

        # regularly show debug information
        if index % 50 == 0:
            self.original_target.test_dnn_model(self.model, "After: Loss of new x: ", x, y)
        return x, y

    def on_epoch_end(self):
        tf.reset_default_graph()

        # debug information at the end of each epoch
        if self.config.enable_optimize:
            print("number of skipped node is: ", self.skipped_node, len(self.y_train))
            self.prev_is_robust = copy.deepcopy(self.is_robust)
            self.is_robust = [False] * len(self.x_original_train)

        print("prediction time is: ", self.predict_time)
        print("selection time is: ", self.total_time)
        
        # reset global info
        self.total_time = 0
        self.predict_time = 0
        self.label_record = [[0 for x in range(self.original_target.num_classes)]
                             for y in range(self.original_target.num_classes)]

        # left-right flip for Cifar-10
        if self.original_target.__class__.__name__ == "Cifar10Model":
            for i in range(len(self.x_original_train)):
                flip = random.choice([True, False])
                if flip:
                    self.x_original_train[i] = np.fliplr(self.x_original_train[i])
            logger.info("Flip done")

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

        elif self.strategy.value == SAU.replace40.value:
            del self.x_train
            temp_x_original_train = copy.deepcopy(self.x_original_train)
            self.x_train, self.y_train = self.au.random40_replace(temp_x_original_train, self.y_train)
            self.x_train = self.original_target.preprocess_original_imgs(self.x_train)
            logger.info(" Augmentation replace40 Done!!!")


    def select_worst(self, x_10=None, y=None):
        """Evaluate the loss of each image, and select the worst one based on loss"""
        ss_time = time.time()
        for i in range(len(x_10)):
            x_10[i] = self.original_target.preprocess_original_imgs(x_10[i])
        num_perturb = len(x_10)

        with self.graph.as_default():
            y_true = np.array(y, dtype='float64')
            loss_all = []
            y_pred_all = []
            
            s_time = time.time()
            for i in range(num_perturb):
                y_pred = self.model.predict_on_batch(x_10[i])
                y_pred = np.clip(np.array(y_pred, dtype='float64'), 1e-8, 1 - 1e-8)
                loss = self.cross_entropy(y_pred, y_true)
                y_pred_all.append(y_pred)
                loss_all.append(loss)
            self.predict_time += time.time() - s_time
        
            x_origin = x_10[0]
            if self.strategy.value == SAU.ga_loss.value:
                y_argmax = np.argsort(loss_all, axis=0)[-2:][::-1]
            else:
                y_argmax = np.argmax(loss_all, axis=0)

            for j in range(len(x_origin)):
                if self.strategy.value == SAU.ga_loss.value:
                    index = y_argmax[0][j]

                    true_label = int(np.argmax(y[j]))
                    predict_label = int(np.argmax(y_pred_all[index][j]))
                    if true_label != predict_label and self.label_record[true_label][predict_label] > 1000:
                        index = y_argmax[1][j]
                        predict_label = np.argmax(y_pred_all[index][j])
                    self.label_record[true_label][predict_label] += 1
                else:
                    index = y_argmax[j]

                if index != 0:
                    x_origin[j] = x_10[index][j]

        self.total_time += time.time() - ss_time
        return x_origin, loss_all

    def optimized_select_worst(self, x_10=None, y=None, is_robust=None):
        """Evaluate the loss of each image, and select the worst one based on loss"""
        ss_time = time.time()
        for i in range(len(x_10)):
            x_10[i] = self.original_target.preprocess_original_imgs(x_10[i])
        x_10 = np.asarray(x_10)

        n = len(x_10[0])
        num_perturb = len(x_10)

        logger.debug("original shape", np.shape(x_10))
        loss_all = []
        predict_all = []
        self.skipped_node += np.sum(is_robust)

        # calculate the loss of each unrobust node
        with self.graph.as_default():
            y_true = np.array(y, dtype='float64')
            unrobust_y = np.array([y[j] for j in range(n) if not is_robust[j]])

            start_time = time.time()
            y_pred = self.model.predict_on_batch(x_10[0])
            self.predict_time += time.time() - start_time

            y_pred = np.clip(np.array(y_pred, dtype='float64'), 1e-8, 1 - 1e-8)
            if self.config.robust_basedon_acc:
                predict = np.argmax(y_pred, axis=1)
                predict_all.append(predict)
            loss = self.cross_entropy(y_pred, y_true)
            loss_all.append(loss)

            # collect the images that should be evaluate
            temp_x = []
            for i in range(num_perturb-1):
                x_10_i = x_10[i+1]
                x_10_i = [x_10_i[j] for j in range(n) if not is_robust[j]]
                temp_x += x_10_i

            # devide the images into batches
            temp_x = np.array(temp_x)
            len_temp_x = len(temp_x)
            y_pred = []
            for i in range(len_temp_x/n):
                start = i*n
                end = start + n
                start_time = time.time()
                y_pred_i = self.model.predict_on_batch(temp_x[start: end])
                self.predict_time += time.time() - start_time
                y_pred += list(y_pred_i)

            remaining = len_temp_x % n
            if remaining != 0:
                start_time = time.time()
                if len_temp_x >= n:
                    y_pred_i = self.model.predict_on_batch(temp_x[-n:])
                else:
                    y_pred_i = self.model.predict_on_batch(temp_x[-remaining:])
                self.predict_time += time.time() - start_time
                y_pred += list(y_pred_i[-remaining:])
       
            num_unrobust_node = len(y_pred) / (num_perturb - 1)
            if num_unrobust_node>0:
                for i in range(num_perturb-1):
                    start = i * num_unrobust_node
                    end = start + num_unrobust_node
                    y_pred_i = np.clip(np.array(y_pred[start:end], dtype='float64'), 1e-8, 1 - 1e-8)
                    if self.config.robust_basedon_acc and len(y_pred_i)>0:
                        predict = np.argmax(y_pred_i, axis=1)
                        predict_all.append(predict)
                    unrobust_loss = self.cross_entropy(y_pred_i, unrobust_y)
                    loss_all.append(unrobust_loss)
            else:
                for i in range(num_perturb-1):
                    loss_all.append([])
                    if self.config.robust_basedon_acc:
                        predict_all.append([])

        # add fake loss for the robust node
        for i in range(n):
            if is_robust[i]:
                for j in range(1, num_perturb):
                    loss_all[j] = np.insert(loss_all[j], i, 0)
                    if self.config.robust_basedon_acc:
                        predict_all[j] = np.insert(predict_all[j], i, predict_all[0][i])


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
        self.total_time += time.time()-ss_time

        predict_acc = [True] * n
        if self.config.robust_basedon_acc:
            labels = np.argmax(y, axis=1)
            for i in range(n):
                for j in range(num_perturb):
                    if predict_all[j][i] != labels[i]:
                        predict_acc[i] = False
                        break

        return x_origin, loss_all, predict_acc


    def generate_cov(self, origin_x=None, x_n=None, y=None):
        """generate loss for each population"""
        for i in range(len(x_n)):
            x_n[i] = self.original_target.preprocess_original_imgs(x_n[i])
        # n = len(x_n[0])
        num_perturb = len(x_n)

        with self.graph.as_default():
            y_true = np.array(y, dtype='float64')
            cov_all = []
            o_1 = self.nc.generate_layer_output(self.original_target.preprocess_original_imgs(origin_x))
            for i in range(num_perturb):
                o_2 = self.nc.generate_layer_output(x_n[i])
                cov = self.nc.compare_output(o_1, o_2)

                y_pred = self.model.predict_on_batch(x_n[i])
                y_pred = np.clip(np.array(y_pred, dtype='float64'), 1e-8, 1 - 1e-8)
                loss = self.cross_entropy(y_pred, y_true)

                cov_all.append(cov+1000*loss)

            x_origin = x_n[0]

            y_argmax = np.argmax(cov_all, axis=0)

            for j in range(len(x_origin)):
                index = y_argmax[j]
                x_origin[j] = x_n[index][j]

        return x_origin, cov_all

