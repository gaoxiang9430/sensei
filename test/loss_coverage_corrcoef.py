from dataset.gtsrb.train import GtsrbModel
from deepaugment.perturbator import Perturbator
from deepaugment.neural_coverage import NeuralCoverage
import copy
import keras
import tensorflow as tf
import numpy as np


def generate_loss(x, y):
    y_predict = model0.predict(x)
    y_true = np.array(y, dtype='float32')

    y_true = tf.convert_to_tensor(y_true)
    y_pred = tf.convert_to_tensor(y_predict)

    loss = keras.losses.categorical_crossentropy(y_true, y_pred)
    loss = keras.backend.get_value(loss)
    return loss

"""
def generate_coverage_diff(model, x, x_perturb):
    coverage_diff = []
    x_layer_outputs = generate_layer_output(model, x)
    x_perturb_layer_outputs = generate_layer_output(model, x_perturb)

    for i in range(len(x)):
        nc = NeuralCoverage(model)
        for layer_index, intermediate_layer_output in enumerate(x_layer_outputs):
            nc.update_neural(layer_index, x_layer_outputs[layer_index][i],
                             coverage_threshold)

        nc2 = NeuralCoverage(model)
        for layer_index, intermediate_layer_output in enumerate(x_perturb_layer_outputs):
            nc2.update_neural(layer_index, x_perturb_layer_outputs[layer_index][i],
                              coverage_threshold)

        diff = nc.compare(nc2)
        coverage_diff.append(diff)
    return coverage_diff


def generate_layer_output(model, input_data):
    layer_names = [layer.name for layer in model.layers if
                   'flatten' not in layer.name and 'input' not in layer.name]

    intermediate_layer_model = Model(inputs=model.input,
                                     outputs=[model.get_layer(layer_name).output
                                              for layer_name in layer_names])
    intermediate_layer_outputs = intermediate_layer_model.predict(input_data)
    return intermediate_layer_outputs
"""

if __name__ == '__main__':
    md = GtsrbModel(source_dir='GTSRB')

    # Load data
    x_original_test, y_original_test = md.load_original_test_data()
    # x_original_test = x_original_test[0:10]
    # y_original_test = y_original_test[0:10]

    x_original_train, y_original_train = md.load_original_data('train')

    # perturb data
    temp_x_test = copy.deepcopy(x_original_test)
    pt = Perturbator()
    x_perturbed_test, y_perturbed_test = pt.random_perturb(temp_x_test, y_original_test)

    # Load model
    _model0 = (0, "models/gtsrb.oxford.model0.hdf5")
    model0 = md.load_model(_model0[0], _model0[1])

    x_original_test = md.preprocess_original_imgs(x_original_test)
    x_perturbed_test = md.preprocess_original_imgs(x_perturbed_test)

    loss_set = generate_loss(x_perturbed_test, y_perturbed_test)
    print loss_set

    nc = NeuralCoverage(model0)
    coverage_diff_set = nc.generate_coverage_diff(x_original_test, x_perturbed_test)
    print coverage_diff_set

    print np.corrcoef(loss_set, coverage_diff_set)

    import matplotlib
    import matplotlib.pyplot as plt

    matplotlib.style.use('ggplot')

    plt.scatter(loss_set, coverage_diff_set)
    plt.show()
