"""
This program is test the implementation of image transformation
Author: Xiang Gao (xiang.gao@us.fujitsu.com)
Time: Sep, 21, 2018
"""

import os

import cv2

import libs.spacial_transformation as tr
from dataset.gtsrb.train import GtsrbModel
from dataset.svhn.train import SVHN
import tensorflow as tf


def image_zoom_test(img=None, param=1.5):
    img = tr.image_zoom(img, param)
    height, width = img.shape[:2]
    print height, width
    return img


def image_brightness_test(img=None, param=128):
    img = tr.image_brightness(img, param)
    return img


def image_blur_test(img=None, param=1.5):
    img = tr.image_blur(img, param)
    return img


def image_contrast_test(img=None, param=1.5):
    img = tr.image_contrast(img, param)
    return img


def image_translate_test(img=None, param=1, param2=2):
    img = tr.image_translation_cropped(img, param, param2)
    return img

def combination_filter(img=None, zoom=1.01, blur=2, contrast=0.75, brightness=32):
    #img = image_zoom_test(img, zoom)
    img = image_blur_test(img, blur)
    img = image_brightness_test(img, brightness)
    img = image_contrast_test(img, contrast)
    return img


def save_img(img=None, name="temp"):
    name = os.path.join(name + '.ppm')
    cv2.imwrite(name, img)


def trans_tf(x, rot):
    # ones = tf.ones(shape=tf.shape(trans_x))
    # zeros = tf.zeros(shape=tf.shape(trans_x))
    # trans = tf.stack([ones,  zeros, -trans_x,
    #                   zeros, ones,  -trans_y,
    #                   zeros, zeros], axis=1)
    import math
    radian = rot * math.pi / 180

    images_ = tf.convert_to_tensor(x, dtype=tf.float32)

    tf_img = tf.contrib.image.rotate(images_, radian)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        rotated_img = sess.run(tf_img)

    # x = tf.contrib.image.rotate(x, rot, interpolation='BILINEAR')
    # x = tf.contrib.image.transform(x, trans, interpolation='BILINEAR')
    # x = tf.image.resize_image_with_crop_or_pad(x, pad_size, pad_size)
    return rotated_img

if __name__ == '__main__':

    md = SVHN()
    x_original_test, y_original_test = md.load_original_test_data()
    img = x_original_test[0]
    print(img)
    img = img*255
    img = img.astype(int)
    print(img)
    save_img(img, "origin")
    img = image_translate_test(x_original_test[0], -1, 1)
    save_img(img, "translate_m1")
    img = image_translate_test(x_original_test[0], 1, 2)
    save_img(img, "translate_1")
    
    img = x_original_test[0]
    save_img(img, "origin")
    img = image_zoom_test(x_original_test[0], 1)
    save_img(img, "zoom_1")
    img = image_zoom_test(x_original_test[0], 1.5)
    save_img(img, "zoom_1.5")
    img = image_zoom_test(x_original_test[0], 2)
    save_img(img, "zoom_2")

    '''
    img = image_blur_test(x_original_test[0], 4)
    save_img(img, "blur_4")
    img = image_blur_test(x_original_test[0], 3)
    save_img(img, "blur_3")
    img = image_blur_test(x_original_test[0], 2)
    save_img(img, "blur_2")
    img = image_blur_test(x_original_test[0], 1)
    save_img(img, "blur_1")
    img = image_blur_test(x_original_test[0], 0)
    save_img(img, "blur_0")
    '''

    img = image_brightness_test(x_original_test[0], -64)
    save_img(img, "brightness__128")
    img = image_brightness_test(x_original_test[0], 128)
    save_img(img, "brightness_128")
    img = image_brightness_test(x_original_test[0], 96)
    save_img(img, "brightness_96")
    img = image_brightness_test(x_original_test[0], 64)
    save_img(img, "brightness_64")
    img = image_brightness_test(x_original_test[0], 32)
    save_img(img, "brightness_32")
    img = image_brightness_test(x_original_test[0], 16)
    save_img(img, "brightness_16")
    img = image_brightness_test(x_original_test[0], 8)
    save_img(img, "brightness_8")
    img = image_brightness_test(x_original_test[0], 4)
    save_img(img, "brightness_4")

    img = image_contrast_test(x_original_test[0], 0)
    save_img(img, "contrast_0")
    img = image_contrast_test(x_original_test[0], 0.4)
    save_img(img, "contrast_0.4")
    img = image_contrast_test(x_original_test[0], 0.8)
    save_img(img, "contrast_0.8")
    img = image_contrast_test(x_original_test[0], 1.2)
    save_img(img, "contrast_1.2")
    img = image_contrast_test(x_original_test[0], 1.6)
    save_img(img, "contrast_1.6")
    img = image_contrast_test(x_original_test[0], 2)
    save_img(img, "contrast_2")

    height, width = x_original_test[0].shape[:2]
    print height, width
    img = tr.image_rotation_cropped(x_original_test[0], 30)
    height, width = img.shape[:2]
    print height, width
    save_img(img, "rotate_30")
    
    # for i in range(0,1):
    #     img = tr.rotate_image(x_original_test[i], -30)
    #     save_img(img, "a"+str(i))
    #
    # import numpy as np
    # x = trans_tf(np.array(x_original_test[0]), np.array([30]))
    # save_img(x_original_test[0], "origin")
    # index = 0
    # save_img(x, str(index))
    # # for i in x:
    # #     index += 1
    # #     save_img(x, str(index))
