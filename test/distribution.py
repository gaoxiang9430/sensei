from dataset.gtsrb.train import GtsrbModel
from deepaugment.neural_coverage import NeuralCoverage
import copy
import tensorflow as tf
import keras
from deepaugment.config import global_config as config
from deepaugment.perturbator import Perturbator
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata
import itertools
from keras import backend as K


def generate_value(model, x, y):

    y_predict = model.predict(x)
    # get_3rd_layer_output = K.function([model.layers[0].input],
    #                                   [model.layers[-1].output])
    # y_predict = get_3rd_layer_output([x])[0]
    y_true1 = np.array(y, dtype='float32')
    y_true1 = tf.convert_to_tensor(y_true1)
    y_pred1 = tf.convert_to_tensor(y_predict)
    value1 = keras.losses.categorical_crossentropy(y_true1, y_pred1)
    return keras.backend.get_value(value1)


def generate_cov(model, origin_x, x, y):
    nc = NeuralCoverage(model)
    o1 = nc.get_layer_output(origin_x)
    o2 = nc.get_layer_output(x)
    diffs = nc.compare_output(o2, o1, y)
    return diffs


def generate_cov2(model, origin_x, x, y, y_original_test):
    nc = NeuralCoverage(model)
    o1 = nc.get_layer_output(origin_x)
    class_cov = nc.generate_cov_for_class(o1, y_original_test)
    o2 = nc.get_layer_output(x)
    # diffs = nc.compare_output(o2, o1, y)
    diffs = nc.compare_class_output(o2, class_cov, y)
    return diffs


def generate_image(img, label):
    pt = Perturbator()
    imgs = []
    labels = []
    r = []
    t = []
    for p1 in config.rotation_range[::1]:
        for p2 in config.translate_range[::1]:
            temp_img = copy.copy(img)
            x_perturbed_test = pt.fix_perturb_img(temp_img, p1, p2)
            imgs.append(x_perturbed_test)
            labels.append(label)
    return imgs, np.array(labels)


def plot(x, y, z, index):
    fig = plt.figure(index)

    ax = fig.gca(projection='3d')
    # Plot the surface.
    surf = ax.plot_surface(x, y, z,  rstride=1, cstride=1, cmap=cm.coolwarm,
                           linewidth=0, antialiased=True)
    # ax.set_zlim(0, 2.21)
    # Customize the z axis.
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

    plt.xlabel('rotate', fontsize=12)
    plt.ylabel('translate', fontsize=12)

    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)


def prepare_data(value):
    xs = np.array(config.rotation_range[::1])
    ys = np.array(config.translate_range[::1])

    # zs = np.array(value).reshape((len(ys), len(xs)))
    # print xs, ys, value
    # xs, ys = np.meshgrid(xs, ys)
    # plot(xs, ys, zs)

    xi = np.linspace(xs.min(), xs.max(), (len(value) / 3))
    yi = np.linspace(ys.min(), ys.max(), (len(value) / 3))

    some_lists = [xs, ys]
    res = []
    for element in itertools.product(*some_lists):
        res.append(list(element))
    x = np.asarray(res)[:, 0]
    y = np.asarray(res)[:, 1]
    z = np.asarray(value)
    print x.shape, y.shape, z.shape
    zi = griddata((x, y), z, (xi[None,:], yi[:,None]), method='cubic')

    zig = np.array(zi).reshape((len(yi), len(xi)))
    xig, yig = np.meshgrid(xi, yi)
    return xig, yig, zig


def save_img(img=None, name="temp"):
    import os
    import cv2
    name = os.path.join(name + '.ppm')
    cv2.imwrite(name, img)


def main():
    md = GtsrbModel(source_dir='GTSRB')
    # _model0 = (0, "models/gtsrbreplace_worst_of_10_model_False.hdf5")
    _model0 = (0, "models/gtsrb.oxford.original_model.hdf5")
    # _model0 = (0, "models/gtsrbga_loss_model_False.hdf5")
    x_original_test, y_original_test = md.load_original_test_data()

    model = md.load_model(_model0[0], _model0[1])

    # index = 50, 10, 20, 30
    index = 10578
    save_img(x_original_test[index], "cur_image")
    img = copy.copy(x_original_test[index])
    label = y_original_test[index]

    x, y = generate_image(img, label)

    temp_x = copy.copy(x)
    value = generate_value(model, md.preprocess_original_imgs(temp_x), y)
    # print value
    xig, yig, zig = prepare_data(value)
    plot(xig, yig, zig, "Loss")

    # value = generate_cov2(model, md.preprocess_original_imgs(x_original_test), md.preprocess_original_imgs(x),
    #                       y, y_original_test)
    # org_imgs = []
    # for i in range(len(y)):
    #     temp_img = copy.copy(img)
    #     org_imgs.append(temp_img)
    # value = generate_cov(model, md.preprocess_original_imgs(org_imgs), md.preprocess_original_imgs(x), y)
    #
    # xig, yig, zig = prepare_data(value)
    # plot(xig, yig, zig, "Cov")
    plt.show()


if __name__ == '__main__':
    main()
