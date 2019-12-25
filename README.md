## Improving the Robustness of Neural Network via Optimized Data Augmentation ##


### Introduction ###

Deep neural networks (DNN) have been shown to be notoriously brittle to small perturbations in their input data. Recently, test generation techniques have been successfully employed to augment existing specifications of intended program behavior, to improve the generalizability of program synthesis and repair. Inspired by these approaches, in this paper, we propose a technique that re-purposes software testing methods, specifically mutation-based fuzzing, to augment the training data of DNNs, with the objective of enhancing their robustness. Our technique casts the DNN data augmentation problem as an optimization problem. It uses genetic search to generate the most suitable variant of an input data to use for training the DNN, while simultaneously identifying opportunities to accelerate training by skipping augmentation in many instances. We instantiate this technique in two tools, SENSEI and SENSEI-SA, and evaluate them on 15 DNN models spanning 5 popular image data-sets. Our evaluation shows that SENSEI can improve the robust accuracy of the DNN, compared to the state of the art, on each of the 15 models. Further, SENSEI-SA can reduce the average DNN training time, while still improving robust accuracy.

### Installation ###
#### required libraries ####
- Tensorflow (or tensonflow-gpu)
- Keras
- Cleverhans
- opencv-python
- imutils
- scikit-learn,scikit-image,pandas (for GSTRB)
- squeezenet


### Run ####
1. Clone Sensei and set up
```
git clone https://github.com/gaoxiang9430/sensei.git
export PYTHONPATH=$PYTHONPATH:`pwd`/sensei/
cd sensei/augment
```
2. Model training (detailed command option can be found in [doc/command.md](./doc/command.md))
```
python augmented_training.py STRATEGY DATASET -m MODEL_ID -e EPOCHs
```
3. Model testing
```
python adversarial_attack.py STRATEGY DATASET -m MODEL_ID
```

