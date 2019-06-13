from augment.perturbator import Perturbator
import os
import cv2
from dataset.gtsrb.train import GtsrbModel
from dataset.cifar10.train_vgg import  Cifar10Model
import copy


def save_img(img=None, name="temp"):
    name = os.path.join(name + '.ppm')
    cv2.imwrite(name, img)


#md = GtsrbModel(source_dir='GTSRB')
md = Cifar10Model()
_model0 = (0, "models/gtsrb.oxford.model0.hdf5")

x_original_test, y_original_test = md.load_original_test_data()

# model0 = md.load_model(_model0[0], _model0[1])

x_part = x_original_test[0:10]
y_part = y_original_test[0:10]

for i in range(len(x_part)):
    save_img(x_part[i], str(i)+"_origin")

pt = Perturbator()
x_part_temp = copy.deepcopy(x_part)
x_perturb, y_perturb = pt.fix_perturb(x_part_temp, y_part, 0, 0, 0.1, 1, 0, 0, 1)
#x_perturb, y_perturb = pt.random_perturb(x_part_temp, y_part)
x_part_temp2 = copy.deepcopy(x_part)
x_perturb2, y_perturb2 = pt.fix_perturb(x_part_temp2, y_part, 0, -3, -0.2, 1, 0, -32, 0.8)
for i in range(len(x_perturb)):
    save_img(x_perturb[i], str(i))
    #save_img(x_perturb2[i], str(i)+"_a")
