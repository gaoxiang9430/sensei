from dataset.gtsrb.train import GtsrbModel
from dataset.cifar10.train import Cifar10Model
from deepaugment.augmentor import Augmenter, Perturbator
from deepaugment.neural_coverage import NeuralCoverage
import copy
import numpy as np
from deepaugment.config import global_config as config


def generate_misclassified_perturbations(model, md, x, y):
    pt = Perturbator()
    for i in range(len(x)):
        x_augs = []
        y_augs = []
        for p1 in config.rotation_range[::20]:
            for p2 in config.translate_range[::2]:
                for p2_v in config.translate_range[::2]:
                    for p3 in config.shear_range[::15]:
                        perturbed_img = pt.fix_perturb_img(copy.deepcopy(x[i]), p1, p2, p2_v, p3)
                        x_augs.append(perturbed_img)
                        y_augs.append(y[i])

        x_augs = md.preprocess_original_imgs(x_augs)
        y_predict = model.predict(x_augs)
        y_predict = np.argmax(y_predict, axis=1)
        y_true = np.argmax(y_augs, axis=1)

        for j in range(len(x_augs)):
            if y_true[j] != y_predict[j]:
                name = str(i)+"_"+str(j)
                print name


def main():
    md = GtsrbModel(source_dir='GTSRB')
    _model0 = (0, "models/gtsrbga_loss_model_False.hdf5")
    #_model0 = (0, "models/gtsrbga_loss_model_False_q2.hdf5")
    #_model0 = (0, "models/gtsrbreplace_worst_of_10_model_False.hdf5")
    x_original_test, y_original_test = md.load_original_test_data()

    model = md.load_model(_model0[0], _model0[1])

    x_part = x_original_test[100:200]
    y_part = y_original_test[100:200]

    generate_misclassified_perturbations(model, md, x_part, y_part)


if __name__ == '__main__':
    main()

