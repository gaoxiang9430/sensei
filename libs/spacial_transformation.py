
from __future__ import print_function

import numpy as np
import cv2
import pickle
import imutils


def hamming2(s1, s2):
    """Calculate the Hamming distance between two bit strings"""
    assert len(s1) == len(s2)
    return sum(c1 != c2 for c1, c2 in zip(s1, s2))


def image_translation_cropped(img, params, params2=0):
    if len(img.shape) == 2:
        rows, cols = img.shape
    else:
        rows, cols, ch = img.shape

    M = np.float32([[1, 0, params], [0, 1, params2]])
    dst = cv2.warpAffine(img, M, (cols, rows))
    return dst


def image_shear_cropped(img, params):
    if len(img.shape) == 2:
        rows, cols = img.shape
    else:
        rows, cols, ch = img.shape
    factor = params*(-1.0)
    M = np.float32([[1, factor, 0], [0, 1, 0]])
    dst = cv2.warpAffine(img, M, (cols, rows))
    if int(factor*cols) == 0:
        return dst
    if params >= 0:
        return dst[:, :int(factor*cols)]
    else:
        return dst[:, int(factor*cols):]


def rotate_image(image, angle):
    image = imutils.rotate_bound(image, angle)
    return image


def image_zoom(image, param):
    """ param: 1-2 """
    res = cv2.resize(image, None, fx=param, fy=param, interpolation=cv2.INTER_LINEAR)
    # res = crop_around_center(res, 32, 32)
    return res


def image_blur(image, params):
    blur = cv2.blur(image, (params+1, params+1))
    return blur


def image_brightness(image, param):
    image = np.int16(image)
    image = image + param
    image = np.clip(image, 0, 255)
    image = np.uint8(image)
    return image


def image_contrast(image, param):
    """
    param: 0-2
    """
    image = np.int16(image)
    image = (image-127) * param + 127
    image = np.clip(image, 0, 255)
    image = np.uint8(image)
    return image


def save_object(obj, filename):
    with open(filename, 'wb') as output:
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)
