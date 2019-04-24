from collections import defaultdict, OrderedDict
from keras.models import Model
from deepaugment.config import *
import multiprocessing
from config import ExperimentalConfig
from keras.backend import int_shape
import tensorflow as tf
import keras
from keras import backend as K


def compare_cov(t1_i, t2_i):
    cov_diff = np.linalg.norm(t1_i-t2_i)
    return cov_diff


class NeuralCoverage:
    def __init__(self, model):
        self.config = ExperimentalConfig.gen_config()
        self.model = model
        self.pool = multiprocessing.Pool(self.config.num_processor)
        layer_name = [layer.name for layer in self.model.layers if
                      'flatten' not in layer.name and 'input' not in layer.name][-2]
        self.intermediate_layer_model = Model(inputs=self.model.input,
                                              outputs=[self.model.get_layer(layer_name).output])

    def generate_layer_output(self, x):
        layer_output = self.intermediate_layer_model.predict_on_batch([x])
        return layer_output

    def compare_output(self, o1, o2):
        coverage_diff = []
        tasks = []
        for index in range(len(o1)):   # number of test
            tasks.append((o1[index], o2[index], ))

        # Run tasks
        results = [self.pool.apply_async(compare_cov, t) for t in tasks]

        for i in range(len(o1)):       # number of test
            diff = results[i].get()
            coverage_diff.append(diff)
        return coverage_diff

