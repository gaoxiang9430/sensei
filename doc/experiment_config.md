### Experiment Configurations

---

#### Supported Perturbation

**Standard perturbation:**

- Rotation: rotate image with degree in range [-30, 30]
- Translate: Shift image at most 10% pixels {GTSRB: [-3,3], Cifar10: [-3,3]}
- Shear: shear image at most 10% pixels [-0.1, 0.1]

**Perturbation based on filter:**

- Zoom: zoom up or zoom down with range [-0.9, 1.1]
- Blur: image blur with range [0,1] (temporarily disabled)
- Brightness: change brightness by uniformly adding or subtracting a value for each pixel, the value is in range [-32, 32] with step 2
- Contrast: change contrast by scale the RGB value of each pixel with a factor in range [0.8, 1.2]

---

**Augmentation strategy:** Standard, Aug.30, Aug.40, Worst-of-10(loss), Worst-of-10(Cov), Genetic(Loss), Genetic(Cov)

- Standard: model trained based on original training data
- Aug.30(40): model trained based on randomly perturbed training data, 30 (40) is the perturbation parameter range
- Worst-of-10: randomly generating 10 perturbations for each image, and train the model using the one with highest loss
- Worst-of-10(Cov): randomly generating 10 perturbations for each image, and train the model using the one with most different coverage
- Genetic(Loss): using genetic algorithm to gurde the the perturbation generation process, the fitness function is based on loss
- Genetic(Coverage): genetic algorithm based on neural coverage (In process)

**Attack:** Nature, Random, Grid (*TODO: fill parameters*)

- Nature: original testing set
- Random: perturb original testing set using random perturbation parameters
- Grid: perturb using grid parameters

