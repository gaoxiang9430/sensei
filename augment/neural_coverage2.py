from collections import defaultdict, OrderedDict
from keras.models import Model
from augment.config import *
import multiprocessing
from config import ExperimentalConfig
from keras.backend import int_shape
import tensorflow as tf
import keras
from keras import backend as K


def scale(intermediate_layer_output):
    min_o = intermediate_layer_output.min()
    x_scaled = (intermediate_layer_output - min_o) / (intermediate_layer_output.max() - min_o)
    return x_scaled


def compare_cov2(t1_i, t2_i, coverage_threshold):
    lower_bound = coverage_threshold*-1
    upper_bound = coverage_threshold
    minus = t1_i - t2_i
    cov_diff = sum(minus > upper_bound) + sum(minus < lower_bound)

    return cov_diff


def compare_cov(t1_i, t2_i):
    cov_diff = np.linalg.norm(t1_i-t2_i)
    return cov_diff


def compare_cov_class(t1_i, t2):
    diff = 0
    value = np.asarray(t1_i) / (0.1 + 1e-7)
    value = value.astype(int)
    for i in range(len(value)):
        if not t2[i * 10 + value[i]]:
            diff += 1
    return diff


def preprocess_cov(output_i):
    preprocessed_output = np.zeros(0)
    for j in range(len(output_i)):  # number of layers
        t1_j = np.asarray(output_i[j])
        t = tuple(range(len(t1_j.shape)-1))
        temp = scale(np.mean(t1_j, axis=t))
        preprocessed_output = np.concatenate((preprocessed_output, temp))

    return np.asarray(preprocessed_output)


class NeuralCoverage:
    def __init__(self, model):
        self.config = ExperimentalConfig.gen_config()
        self.model = model
        # self.model_layer_dict = OrderedDict()
        # for layer in model.layers:
        #     if 'flatten' in layer.name or 'input' in layer.name:
        #         continue
        #
        #     for index in range(layer.output_shape[-1]):
        #         self.model_layer_dict[(layer.name, index)] = False
        self.layer_names = [layer.name for layer in model.layers if
                            'flatten' not in layer.name and 'input' not in layer.name][-2:]

        self.intermediate_layer_model = Model(inputs=model.input,
                                              outputs=[model.get_layer(layer_name).output
                                                       for layer_name in self.layer_names])
        self.pool = multiprocessing.Pool(self.config.num_processor)
        self.class_size = int_shape(model.get_layer(self.layer_names[-1]).output)[-1]

    def neuron_covered(self):
        covered_neurons = len([v for v in self.model_layer_dict.values() if v])
        total_neurons = len(self.model_layer_dict)
        return covered_neurons, total_neurons, covered_neurons / float(total_neurons)

    def update_coverage(self, input_data):
        """update neural coverage based on input and return the number of newly covered neural"""
        intermediate_layer_outputs = self.intermediate_layer_model.predict_on_batch(input_data)
        for j in range(len(intermediate_layer_outputs[0])):
            for i, intermediate_layer_output in enumerate(intermediate_layer_outputs):
                self.update_neural(i, intermediate_layer_output[j])

    def update_neural(self, layer_index, intermediate_layer_output):
        """ update neural based on output """
        scaled = scale(intermediate_layer_output)
        for num_neuron in xrange(scaled.shape[-1]):
            if np.mean(scaled[..., num_neuron]) > self.config.coverage_threshold:
                self.model_layer_dict[(self.layer_names[layer_index], num_neuron)] = True

    def full_coverage(self):
        if False in self.model_layer_dict.values():
            return False
        return True

    def get_num_neural(self):
        return len(self.model_layer_dict.values())

    def generate_coverage_dict(self, x):
        model_layer_dicts = []
        x_layer_outputs = self.intermediate_layer_model.predict(x)
        for i in range(len(x)):
            # clear neural previous neural coverage
            self.model_layer_dict = self.model_layer_dict.fromkeys(self.model_layer_dict, False)
            for layer_index, intermediate_layer_output in enumerate(x_layer_outputs):
                self.update_neural(layer_index, x_layer_outputs[layer_index][i])
            model_layer_dicts.append(self.model_layer_dict.values())
        return model_layer_dicts

    def get_layer_output(self, x):
        # output = self.intermediate_layer_model.predict_on_batch(x)
        # output = list(zip(*output))
        # tasks = []
        # preprocessed_outputs = []
        # for index in range(len(output)):   # number of test
        #     # preprocessed_outputs.append(preprocess_cov(output[index]))
        #     tasks.append((output[index],))
        # # Run tasks
        # results = [self.pool.apply_async(preprocess_cov, t) for t in tasks]
        # for i in range(len(results)):       # number of test
        #     preprocessed_output = results[i].get()
        #     preprocessed_outputs.append(preprocessed_output)
        # del output[:]
        # del output
        preprocessed_outputs = np.array(self.model.predict(x), dtype='float64')
        return preprocessed_outputs

    # counter coverage for class according to the confidence of prediction
    def generate_cov_for_class(self, x, y):
        class_cov = dict()
        for i in range(self.class_size):
            class_cov[i] = np.zeros(10*len(x[0]), dtype=bool)

        y_predict_temp = np.asarray(x)[:, -1 * self.class_size:]

        predicts = np.argmax(y_predict_temp, axis=1)
        y_true = np.argmax(y, axis=1)
        for j in range(len(x)):
            # ignore this test if it is misclassified
            if predicts[j] != y_true[j]:
                continue
            predict = predicts[j]
            value = np.asarray(x[j])/(0.1+1e-7)
            value = value.astype(int)
            for k in range(len(value)):
                class_cov[predict][k*10+value[k]] = True
        # for i in range(self.class_size):
        #     print(class_cov[i].sum())

        return class_cov

    def compare_class_output(self, o1, class_cov, y):
        labels = np.argmax(y, axis=1)
        coverage_diff = []
        tasks = []
        for index in range(len(o1)):   # number of test
            tasks.append((o1[index], class_cov[labels[index]], ))

        # Run tasks
        results = [self.pool.apply_async(compare_cov_class, t) for t in tasks]

        y_predict_temp = np.asarray(o1)[:, -1 * self.class_size:]
        y_true = np.array(y, dtype='float64')
        y_true = tf.convert_to_tensor(y_true)
        y_pred = tf.convert_to_tensor(y_predict_temp)

        loss = keras.losses.categorical_crossentropy(y_true, y_pred)
        loss = keras.backend.get_value(loss)

        for i in range(len(o1)):       # number of test
            diff = results[i].get()
            # coverage_diff.append(diff)
            if loss[i] < 1e-3:
                diff = 0
            coverage_diff.append(diff + loss[i])

        return coverage_diff

    def compare_output(self, o1, o2, y):
        coverage_diff = []
        print ("length of o1 is : ", len(o1[0]))
        # o1 = zip(*o1)
        # o2 = zip(*o2)
        tasks = []
        for index in range(len(o1)):   # number of test
            tasks.append((o1[index][:-1 * self.class_size], o2[index][: -1 * self.class_size], ))

        # Run tasks
        results = [self.pool.apply_async(compare_cov, t) for t in tasks]

        y_predict_temp = np.asarray(o1)[:, -1 * self.class_size:]
        y_true = np.array(y, dtype='float64')
        y_true = tf.convert_to_tensor(y_true)
        y_pred = tf.convert_to_tensor(y_predict_temp)

        loss = keras.losses.categorical_crossentropy(y_true, y_pred)
        loss = keras.backend.get_value(loss)

        for i in range(len(o1)):       # number of test
            diff = results[i].get()
            # coverage_diff.append(diff + loss[i]*100)
            # coverage_diff.append(diff+(int(math.log(loss[i])) + 50) * 50)
            if loss[i] < 1e-2:
                diff = 0
            coverage_diff.append(diff + 10 * loss[i])
        return coverage_diff

    def compare_output_sub(self, o1, o2, loss):
        coverage_diff = []
        tasks = []
        for index in range(len(o1)):   # number of test
            tasks.append((o1[index], o2[index], ))

        # Run tasks
        results = [self.pool.apply_async(compare_cov, t) for t in tasks]

        for i in range(len(o1)):       # number of test
            diff = results[i].get()
            if loss[i] < 1e-3:
                diff = 0
            coverage_diff.append(diff + 100 * loss[i])
        return coverage_diff

    def compare_output2(self, x_pre, y, n):

        y_predict_temp = self.model.predict_on_batch(x_pre[n:])

        y_true1 = np.array(y, dtype='float32')
        y_true1 = tf.convert_to_tensor(y_true1)
        y_pred1 = tf.convert_to_tensor(y_predict_temp)
        loss1 = keras.losses.categorical_crossentropy(y_true1, y_pred1)
        loss = keras.backend.get_value(loss1)

        get_3rd_layer_output = K.function([self.model.layers[0].input],
                                          [self.model.layers[-2].output])
        layer_output = get_3rd_layer_output([x_pre])
        origin_cov = layer_output[:n]
        for i in range(n, len(layer_output), n):
            cov = layer_output[i:i+n]
            loss[i - n, n] = self.compare_output_sub(origin_cov, cov, loss[i-n, n])

        return loss

    def generate_coverage_diff(self, x, x2):
        x_model_layer_dicts = self.generate_coverage_dict(x)
        x2_model_layer_dicts = self.generate_coverage_dict(x2)
        return self.compare_dicts(x_model_layer_dicts, x2_model_layer_dicts)

    def compare_dicts(self, d1, d2):
        coverage_diff = []
        for i in range(len(d1)):
            model_layer_dict1 = d1[i]
            model_layer_dict2 = d2[i]
            # count the difference between two coverage set
            diff = np.count_nonzero(np.logical_xor(model_layer_dict1, model_layer_dict2))
            coverage_diff.append(diff)
        return coverage_diff

    def merge_coverage_dict(self, d1, d2):
        for i in range(len(d1)):
            d1[i] = list(np.logical_or(d1[i], d2[i]))
        return d1

    """
    from keras.models import Model
    def fired(self, model, layer_name, index, input_data, threshold=0):
        intermediate_layer_model = Model(inputs=model.input, outputs=model.get_layer(layer_name).output)
        intermediate_layer_output = intermediate_layer_model.predict(input_data)[0]
        scaled = self.scale(intermediate_layer_output)
        if np.mean(scaled[..., index]) > threshold:
            return True
        return False
    """
