## Experimental Results 
### Dataset: German Traffic Sign Benchmarks (GTSRB) with 50000 images and 43 labels
Model: Oxford model

<!---
**only with standard perturbation for each image (rotation, translation, shear):**

|          | Natural |Random | Grid  |
|--------- | ------- | ----- | ----- |
| Standard | 0.973   | 0.742 | 0.067 |
| Aug.30   | 0.976   | 0.965 | 0.757 |
| Aug.40   | 0.971   | 0.966 | 0.789 |
| W-10     | 0.973   | 0.963 | 0.803 |
| W-10(O1) | 0.973   | 0.963 | 0.801 |
| W-10(Cov)| 0.981   | 0.969 | 0.832 |
| GA(Loss) | 0.981   | 0.976 | 0.877 |
| GA(Cov)  | 0.981   | 0.980 | 0.887 |
-->

####batch-level augmentation 
with rotation(-30, 30), translate(-3,3), shear(-0.1, 0.1)

|          | Natural |Random | Grid  | Rotation | Translate | Shear |
|--------: | ------: | -----: | ----: |------: | -------: | ------: | 
| Standard | 0.979   | 0.662 | 0.007 | 0.122    | 0.404    |  0.939  |
| Aug.30   | 0.983   | 0.978 | 0.759 | 0.901    | 0.864    | 0.947   |
| Aug.40   | 0.978   | 0.977 | 0.795 | 0.909    | 0.846    | 0.941   |
| W-10     | 0.980   | 0.976 | 0.837 | 0.918    | 0.906    | 0.946   |
| W-10(Cov)| 0.983   | 0.980 | 0.845 | 
| GA(Loss) | 0.986   | 0.983 | 0.888 | 0.925    | 0.921    | 0.958   |

**Average number of misclassified perturbations (totally 81 for each image)**

|       | Standard | Aug.30 | Aug.40 | W-10 | GA(loss) |
|------ | -----:   |   ---: | -----: | ---: | -----: |
| #fail | 47.2     | 3.1    | 2.67   | 2.66 | 1.87   |

**with extended perturbation based on filter(zoom, contrast, brighness):**

|          | Natural | Random | Grid  |Rotation | Translate | Shear |
|--------- | ------: | -----: | ----: |------: | -------: | ------: |
| Standard | 0.973   | 0.586  | 0.067 | 0.123   | 0.404    | 0.940   |
| Aug.30   | 0.979   | 0.956  | 0.430 | 0.901   | 0.888    | 0.937   |
| W-10     | 0.985   | 0.961  | 0.586 | 0.932   | 0.922    | 0.960   |
| GA(Loss) | 0.988   | 0.972  | 0.673 | 0.942  |  0.942   | 0.963   |

**Average number of misclassified perturbations(totally 2187 for each image)**

|       | Standard | Aug.30 | W-10 | GA(loss) |
|------ | -----:   |   ---: | ---: | -------: |
| #fail | 789      | 164    | 117  | 98       |

---

#### Dataset: Cifar 10 Benchmarks (Cifar10) with 60000 images and 10 labels
Model: Resnet 20

<!---
**only with standard perturbation for each image (rotation, translation, shear):**

|          | Natural |Random | Grid  |
|--------- | ------- | ----- | ----- |
| Standard | 0.877   | 0.505 | 0.030 |
| Aug.30   | 0.892   | 0.884 | 0.604 |
| Aug.40   | 0.872   | 0.891 | 0.626 |
| W-10     | 0.874   | 0.863 | 0.602 |
| GA(Loss) | 0.903   | 0.889 | 0.700 |
| GA(Cov)  |
-->

**batch-level augmentation**

|          | Natural | Random | Grid |Rotation | Translate| Shear   |
|--------- | ------: | -----: | ----:|------: | -------: | ------: |
| Standard | 0.875   | 0.496  | 0.013| 0.093   | 0.202    | 0.494   |
| Aug.30   | 0.897   | 0.897  | 0.522| 0.697   | 0.651    | 0.788   |
| Aug.40   | 0.872   | 0.891  | 0.626|
| W-10     | 0.892   | 0.893  | 0.686| 0.738   | 0.701    | 0.811   |
| GA(Loss) | 0.915   | 0.913  | 0.732| 

**Average number of misclassified perturbations (totally 81 for each image)**

|       | Standard | Aug.30 | W-10 | GA(loss) |
|------ | -----:   |   ---: | ---: | -------: |
| #fail | 45.0     | 10.4   | 9.8  | 8.4      |

**with extended perturbation based on filter(zoom, contrast, brighness):**

|          | Natural | Random | Grid |Rotation | Translate| Shear   |
|--------- | ------: | -----: | ----:|------: | -------: | ------: |
| Standard | 0.890   | 0.435  | 0.011| 0.100   | 0.220    | 0.530   |
| Aug.30   | 0.901   | 0.892  | 0.352| 0.732   |0.683     | 0.807   |
| W-10     | 0.912   | 0.903  | 0.469| 0.769   | 0.734    | 0.817   |
| GA(Loss) | 0.915   | 0.913  | 0.560| 0.786   | 0.740    | 0.841   |

**Average number of misclassified perturbations(totally 2187 for each image)**

|       | Standard | Aug.30 | W-10 | GA(loss) |
|------ | -----:   |   ---: | ---: | -------: |
| #fail | 1316     | 263    | 241  | 232      |

