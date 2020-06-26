The command option for model training:

Note that this repo only contains the code to train models, but does **not** include the original dataset. Since the datasets of cifar10 and fashionmnist are included in the kares library, you can directly train their models. As for the other dataset, you first need to manually download the dataset and put it in the dataset/dataset-name directory.

```
usage: augmented_training.py [-h] [-q QUEUE] [-m MODEL] [-t START_POINT]
                             [-r THRESHOLD] [-e EPOCH] [-f] [-o]
                             strategy dataset

positional arguments:
  strategy              augmentation strategy, supported strategy:['original', 'replace30',
                        'replace40', 'replace_worst_of_10', 'ga_loss (Sensei)', 'ga_cov']
  dataset               the name of dataset, support dataset:['gtsrb', 'cifar10',
                        'fashionmnist', 'svhn', 'imdb', 'utk', 'kvasir']

optional arguments:
  -h, --help            show this help message and exit
  -q QUEUE, --queue QUEUE
                        the length of queue for genetic algorithm (default 10)
  -m MODEL, --model MODEL
                        selection of model
  -t START_POINT, --start-point START_POINT
                        the start point of epoch (default from epoch 0)
  -r THRESHOLD, --threshold THRESHOLD
                        the loss threshold for selective augmentation (default 1e-3)
  -e EPOCH, --epoch EPOCH
                        the number of training epochs (default 30)
  -f, --filter          enable filter transformation operators (zoom, contrast, brightness)
  -o, --optimize        enable selective augmentation
```

The command option for model testing:
```
usage: adversarial_attack.py [-h] [-f] [-o] [-m MODEL] strategy dataset

positional arguments:
  strategy              augmentation strategy, supported strategy:['original', 'replace30',
                        'replace40', 'replace_worst_of_10', 'ga_loss (Sensei)', 'ga_cov']
  dataset               the name of dataset, support dataset:['gtsrb', 'cifar10'
                        fashionmnist', 'svhn', 'imdb', 'utk', 'kvasir']

optional arguments:
  -h, --help            show this help message and exit
  -f, --filter          enable filter transformation operators (zoom, contrast, brightness)
  -o, --optimize        enable optimize
  -m MODEL, --model MODEL
                        selection of model
```
