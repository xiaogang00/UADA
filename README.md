# Universal Adaptive Data Augmentation

This is a pytorch project for the paper [**Universal Adaptive Data Augmentation**] by Xiaogang Xu, Hengshuang Zhao, and Philip Torr.

## Introduction
Existing automatic data augmentation (DA) methods either ignore updating DA's parameters according to the target model's state during training or adopt update strategies that are not effective enough.
In this work, we design a novel data augmentation strategy called "Universal Adaptive Data Augmentation" (UADA). Different from existing methods, UADA would adaptively update DA's parameters according to the target model's gradient information during training: 
given a pre-defined set of DA operations, we randomly decide types and magnitudes of DA operations for every data batch during training, and adaptively update DA's parameters along the gradient direction of the loss concerning DA's parameters. 
In this way, UADA can increase the training loss of the target networks, and the target networks would learn features from harder samples to improve the generalization.

<img src="./figure/framework.pdf" width="900"/>

[paper link](https://arxiv.org/abs/2207.06658)

## Project Setup

First install Python 3. We advise you to install Python 3 and PyTorch with Anaconda:

```
conda create --name py36 python=3.6
source activate py36
```

Clone the repo and install the complementary requirements:
```
cd $HOME
git clone --recursive git@github.com:xiaogang00/UADA.git
cd ddcat
pip install -r requirements.txt
```

The environment of our experiments is CUDA10.2 and TITAN V.

## Requirement

- Hardware: 1 GPU (better with >=11G GPU memory)

## Train

- Download related datasets, e.g., CIFAR10, CIFAR100, and TinyImageNet for image classification.
- Modify the data path for CIFAR10 by modifying "root" in [cifar_noaug.py](data/cifar_noaug.py.py)
- Modify the data path for CIFAR100 by modifying "root" in [cifar100_noaug.py](data/cifar100_noaug.py)
- Modify the data path for TinyImageNet by modifying "train_dir" and "test_dir" in [tiny_imagenet_noaug.py](data/tiny_imagenet_noaug.py)

### Image Classification

#### CIFAR10

- Train the model on CIFAR10 with WideResNet 28-10 or ResNet18 as
  ```
  python train_CIFAR10.py --model-dir ./output/model_CIFAR10 --epochs 300
  ```

#### CIFAR100

- Train the model on CIFAR100 with WideResNet 28-10 or ResNet18 as
  ```
  python train_CIFAR100.py --model-dir ./output/model_CIFAR100 --epochs 300
  ```

#### TinyImageNet

- Train the model on TinyImageNet with ResNet18 or ResNet50 as
  ```
  python train_tinyimagenet.py --model-dir ./output/model_TinyImageNet --epochs 200
  ```

During the training, you can also use --epoch_resume for training resume.

## Test

### Image Classification

#### CIFAR10

- Test the model on CIFAR10 with
  ```
  python test_CIFAR10.py --model_dir ./output/model_CIFAR10
  ```

#### CIFAR100

- Test the model on CIFAR100 with
  ```
  python test_CIFAR100.py --model_dir ./output/model_CIFAR10
  ```

#### TinyImageNet

- Test the model on TinyImageNet with
  ```
  python test_tinyimagenet.py --model_dir ./output/model_TinyImageNet
  ```

## Citation Information

If you find the project useful, please cite:

```
@article{xu2022uada,
  title={Universal Adaptive Data Augmentation},
  author={Xiaogang Xu, Hengshuang Zhao, and Philip Torr},
  journal={arxiv},
  year={2022}
}
```


## Acknowledgments
This source code is inspired by [sam](https://github.com/davda54/sam).

## Contributions
If you have any questions/comments/bug reports, feel free to e-mail the author Xiaogang Xu ([xiaogangxu00@gmail.com](xiaogangxu00@gmail.com)).
