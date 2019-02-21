from dataset.cifar10.train_vgg import Cifar10Model
from deepaugment.augmentor import Augmenter
from deepaugment.neural_coverage import NeuralCoverage
import copy
import numpy as np
import os
import cv2
import tensorflow as tf
import keras


def save_img(img=None, name="temp"):
    name = os.path.join(name + '.ppm')
    cv2.imwrite(name, img)


def rule_out_misclassified_images(model, md, x_part, y_part):
    y_true = np.argmax(y_part, axis=1)
    origin_temp = copy.copy(x_part)
    x_part_new = []
    y_part_new = []
    y_predict = np.argmax(model.predict(md.preprocess_original_imgs(origin_temp)), axis=1)
    for i in range(len(y_predict)):
        if y_predict[i] == y_true[i]:
            x_part_new.append(x_part[i])
            y_part_new.append(y_part[i])

    x_part = x_part_new
    y_part = y_part_new

    print "len: ", len(x_part), len(y_part)
    return x_part, y_part


def save_original_image(x_part, y_part):
    y_true = np.argmax(y_part, axis=1)
    for i in range(43):
        for j in range(len(y_true)):
            if y_true[j] == i:
                save_img(x_part[j], str(i))
                break


def get_misclassified_perturbations(model, md, x_part, y_part):
    au = Augmenter()
    temp_x_original_train = copy.deepcopy(x_part)
    x_aug, y_train = au.worst_of_10(temp_x_original_train, y_part)

    x_aug_after_preprocess = []
    for i in range(len(x_aug)):
        temp = copy.deepcopy(x_aug[i])
        x_aug_after_preprocess.append(md.preprocess_original_imgs(temp))

    fault_index = 0
    nc = NeuralCoverage(model)
    o1 = nc.get_layer_output(md.preprocess_original_imgs(x_part))
    y_true = np.argmax(y_part, axis=1)
    for i in range(10):
        print "========================= Round ", i, "============================"
        set_i = np.array(x_aug_after_preprocess)[:, i]
        y_predict = model.predict(set_i)

        y_true1 = np.array(y_part, dtype='float32')
        y_true1 = tf.convert_to_tensor(y_true1)
        y_pred1 = tf.convert_to_tensor(y_predict)
        loss1 = keras.losses.categorical_crossentropy(y_true1, y_pred1)
        loss1 = keras.backend.get_value(loss1)

        j = 745
        print loss1[j]
        y_predict = np.argmax(y_predict, axis=1)

        o2 = nc.get_layer_output(set_i)
        diffs = nc.compare_output(o2, o1, y_part)
        print "coverage:",diffs[j]

        print "true_"+str(y_true[j])+"_pred_"+str(y_predict[j])

        # y_predict = np.argmax(y_predict, axis=1)
        # for j in range(len(set_i)):
        #     if y_true[j] != y_predict[j]:
        #         fault_index += 1
        #         name = "true_"+str(y_true[j])+"_pred_"+str(y_predict[j])+"_"+str(fault_index)+"_"+str(j)
        #         print name
        #         save_img(x_part[j], name+"_org")
        #         save_img(x_aug[j][i], name)


def main():
    md = Cifar10Model()
    _model0 = (0, "models/cifar10ga_loss_model_False.hdf5")
    x_original_test, y_original_test = md.load_original_test_data()

    model = md.load_model(_model0[0], _model0[1])

    x_part = x_original_test[0:1000]
    y_part = y_original_test[0:1000]

    # save_original_image(x_part, y_part)

    x_part, y_part = rule_out_misclassified_images(model, md, x_part, y_part)

    get_misclassified_perturbations(model, md, x_part, y_part)


if __name__ == '__main__':
    main()
