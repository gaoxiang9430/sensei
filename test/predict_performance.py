from dataset.gtsrb.train import GtsrbModel
from dataset.cifar10.train import Cifar10Model
from deepaugment.augmentor import Augmenter, Perturbator
import copy
import numpy as np
from deepaugment.config import global_config as config
import time
import tensorflow as tf
import keras
import sklearn


def cross_entropy(predictions, targets):
    ce = -np.sum(targets*np.log(predictions)+(1-targets)*np.log(1-predictions), axis=1)
    return ce

def performance(model, md, x, y):
    pt = Perturbator()
    for i in range(1000):
        x_augs, y = pt.random_perturb(copy.deepcopy(x), y)
        x_augs = md.preprocess_original_imgs(x_augs)
        start_time = time.time()

        y_pred1 = np.clip(np.array(model.predict(x_augs), dtype='float64'), 1e-8, 1-1e-8)
        #y_true1 = np.clip(np.array(y, dtype='float64'), 1e-8, 1-1e-8)
        y_true1 = np.array(y, dtype='float64')
        cross_entropy(y_pred1, y_true1)

        '''
        y_pred1 = tf.convert_to_tensor(y_pred1)
        y_true1 = tf.convert_to_tensor(y_true1)
        
        loss1 = keras.losses.categorical_crossentropy(y_true1, y_pred1)
        loss = keras.backend.get_value(loss1)
        print loss
        '''
        print("prediction time:", i, time.time() - start_time)

def main():
    print cross_entropy(np.array([[0.4, 0.6],[0.6,0.4]]), np.array([[1,0],[0,1]]))
    
    md = GtsrbModel(source_dir='GTSRB')
    _model0 = (0, "models/gtsrbga_loss_model_False.hdf5")
    #_model0 = (0, "models/gtsrbga_loss_model_False_q2.hdf5")
    #_model0 = (0, "models/gtsrbreplace_worst_of_10_model_False.hdf5")
    x_original_test, y_original_test = md.load_original_test_data()

    model = md.load_model(_model0[0], _model0[1])

    x_part = x_original_test[1:1000]
    y_part = y_original_test[1:1000]

    performance(model, md, x_part, y_part)
    


if __name__ == '__main__':
    main()

