## Time Table in Minutes(m)


|Estimated training time| GTSRB  | Cifar10 |
|-------------          | ------ | ------- |
| Original              | 18.5   |
| Aug.30                | 27.5   |
| Worst of 10 (Loss)    | 181    | 950     |
| Worst of 10 (Cov)     | 340    |
| Genetic (Loss)        | 181    |
| Genetic (Cov)         |        |

- Training time is collect based on the GPU Server (56 cores, Tesla K80)
- The training of GA_loss is based on queue length 10.
- Coverage oriented approaches are collected based on 56 processes.
