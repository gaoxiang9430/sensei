from dataset.gtsrb.train import GtsrbModel
from deepaugment.neural_coverage import NeuralCoverage
import time
from deepaugment.augmentor import Augmenter
import copy
import gc
import numpy as np
import tensorflow as tf
import keras


def function_test(model, x1, x2, y):
    nc = NeuralCoverage(model)

    start_time = time.time()
    o1 = nc.get_layer_output(x1)
    class_cov = nc.generate_cov_for_class(o1, y)

    o2 = nc.get_layer_output(x2)
    diffs = nc.compare_class_output(o2, class_cov, y)
    print("--- %s seconds ---" % (time.time() - start_time))

    y_true1 = np.array(y, dtype='float32')
    y_true1 = tf.convert_to_tensor(y_true1)
    y_pred1 = tf.convert_to_tensor(model.predict(x2))
    loss1 = keras.losses.categorical_crossentropy(y_true1, y_pred1)
    loss1 = keras.backend.get_value(loss1)

    for i in range(len(diffs)):
        if loss1[i] > 1e-2:
            print loss1[i], diffs[i]
    print("=====================================================")

    same = 0
    diff = 0
    same_sum = 0
    diff_sum = 0
    y2_predict = np.argmax(model.predict(x2), axis=1)
    y_true = np.argmax(y, axis=1)

    for i in range(len(x1)):
        if y_true[i] == y2_predict[i]:
            same += 1
            same_sum += diffs[i]
        else:
            diff += 1
            diff_sum += diffs[i]
            print loss1[i], diffs[i]
        # print y_true[i] == y2_predict[i], diffs[i]

    print diff_sum / float(diff)
    print same_sum / float(same)


def select_worst_cov(md, model, original_x=None, x_10=None, y=None):
    nc = NeuralCoverage(model)
    """select worst image based on neural coverage"""
    cov_diff = []
    loss = []
    for i in range(len(x_10)):
        x_10[i] = md.preprocess_original_imgs(x_10[i])
    origin_cov = nc.get_layer_output(original_x)
    # class_cov = nc.generate_cov_for_class(origin_cov, y)

    # del origin_cov[:]
    # del origin_cov

    print np.argmax(y, axis=1)
    for i in range(10):
        set_i = np.array(x_10)[:, i]
        cov = nc.get_layer_output(set_i)
        # cov_diff_i = nc.compare_class_output(cov, class_cov, y)
        cov_diff_i = nc.compare_output(cov, origin_cov, y)
        del cov[:]
        del cov
        cov_diff.append(cov_diff_i)

        y_predict_temp = model.predict(set_i)
        y_true1 = np.array(y, dtype='float32')

        y_true1 = tf.convert_to_tensor(y_true1)
        y_pred1 = tf.convert_to_tensor(y_predict_temp)

        def categorical_crossentropy_wrapper(y_true, y_pred):
            y_pred = keras.backend.clip(y_pred, 1e-8, 1-1e-8)
            return keras.losses.categorical_crossentropy(y_true, y_pred)

        # loss1 = keras.losses.categorical_crossentropy(y_true1, y_pred1)
        loss1 = categorical_crossentropy_wrapper(y_true1, y_pred1)
        loss1 = keras.backend.get_value(loss1)
        loss.append(loss1)

        print np.argmax(y_predict_temp, axis=1)
        print cov_diff_i
        print loss1

    y_argmax = np.argmax(cov_diff, axis=0)
    y_argmax_loss = np.argmax(loss, axis=0)
    for j in range(len(y_argmax)):
        print y_argmax[j], y_argmax_loss[j]


def memory_test(original_target=None, model=None, x=None, x_10=None):
    """select worst image based on neural coverage"""
    nc = NeuralCoverage(model)
    cov_diff = []

    for i in range(len(x_10)):
        x_10[i] = original_target.preprocess_original_imgs(x_10[i])
    x = original_target.preprocess_original_imgs(x)
    origin_cov = nc.get_layer_output(x)

    for i in range(10):
        set_i = np.array(x_10)[:, i]
        start_time = time.time()
        cov = nc.get_layer_output(set_i)
        print("--- first %s seconds ---" % (time.time() - start_time))
        cov_diff_i = nc.compare_output(origin_cov, cov)
        print("--- %s seconds ---" % (time.time() - start_time))

        del cov[:]
        del cov
        cov_diff.append(cov_diff_i)
    y_argmax = np.argmax(cov_diff, axis=0)
    print y_argmax
    del origin_cov[:]
    del origin_cov
    for j in range(len(y_argmax)):
        index = y_argmax[j]
        x[j] = x_10[j][index]  # update x_train (not good design)
    return x


def main():
    md = GtsrbModel(source_dir='GTSRB')
    _model0 = (0, "models/gtsrbga_loss_model_False.hdf5")
    x_original_test, y_original_test = md.load_original_test_data()

    model0 = md.load_model(_model0[0], _model0[1])

    x_part = x_original_test[0:2000]
    y_part = y_original_test[0:2000]

    au = Augmenter()
    temp_x_original_train = copy.deepcopy(x_part)
    x_part2, y_part = au.random_replace(temp_x_original_train, y_part)

    x_part = md.preprocess_original_imgs(x_part)
    x_part2 = md.preprocess_original_imgs(x_part2)

    function_test(model0, x_part, x_part2, y_part)


def main2():
    md = GtsrbModel(source_dir='GTSRB')
    _model0 = (0, "models/gtsrbga_loss_model_False.hdf5")
    x_original_test, y_original_test = md.load_original_test_data()

    model0 = md.load_model(_model0[0], _model0[1])

    x_part = x_original_test[0:1000]
    y_part = y_original_test[0:1000]

    au = Augmenter()
    temp_x_original_train = copy.deepcopy(x_part)

    x_10, y_train = au.worst_of_10(temp_x_original_train, y_part)
    select_worst_cov(md, model0, md.preprocess_original_imgs(temp_x_original_train), x_10, y_train)

    # x_aug, y_train = au.random_replace(temp_x_original_train, y_part)
    # function_test(model0, md.preprocess_original_imgs(x_part), md.preprocess_original_imgs(x_aug), y_part)


def main3():
    md = GtsrbModel(source_dir='GTSRB')
    _model0 = (0, "models/gtsrbreplace_worst_of_10_cov_model_False.hdf5")
    x_original_test, y_original_test = md.load_original_test_data()

    model0 = md.load_model(_model0[0], _model0[1])

    x_part = x_original_test[0:32]
    y_part = y_original_test[0:32]

    au = Augmenter()
    start_time = time.time()
    temp_x_original_train = copy.deepcopy(x_part)
    x_aug, y_train = au.worst_of_10(temp_x_original_train, y_part)
    print("augmentation done")
    print("--- %s seconds ---" % (time.time() - start_time))
    memory_test(md, model0, x_part, x_aug)
    print("--- %s seconds ---" % (time.time() - start_time))
    gc.collect()

if __name__ == '__main__':
    start_time = time.time()
    main()
