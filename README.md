AutoLoss: Learning Discrete Schedules for Alternate Optimization
======================
Code for reproducing experiments in [AutoLoss: Learning Discrete Schedules for Alternate Optimization](https://arxiv.org/abs/1810.02442).

## Requirements
```
python>=3.6, tensorflow-gpu==1.8.0, matplotlib==2.2.2
```

## Getting Started

### Install and Activate a Virtual Environment
```
virtualenv --python=python3.6 ./env
source ./env/bin/activate
```
### Install Dependencies
```
pip install -r requirements.txt
```

## Datasets
### Regression
```
cd ./dataio
python dataGeneration_reg.py
cd ..
```

### Classification
```
cd ./dataio
python dataGeneration_cls.py
cd ..
```

### GANs
The MNIST database is available at [yann.lecun.com/exdb/mnist/](yan.lecun.com/exdb/mnist).
You just need to set a path to the argument 'data\_dir' in config file 'config/gan/cfg.py'. The database will be downloaded to that folder at the first run.

The Cifar10 database is available at [Download page of CIFAR10](http://www.cs.toronto.edu/~kriz/cifar.html)
You need to download the python version from this page and unzip it. Set the argument 'data\_dir' in config file 'config/gan_cifar.py' to where you save the data.

### Multi-task Neural Translation
The preprocessing of corpus of three language tasks is cumbersome, you can directly use the preprocessed data we provided in this repository.
Or you can run:
```
cd ./dataio
python dataGeneration_mnt.py
cd ..
```
In this case, you need to download 'tiger\_release\_aug07.corrected.16012013.xml' from [download page of the TIGER corpus](http://www.ims.uni-stuttgart.de/forschung/ressourcen/korpora/tiger.en.html) and save it at 'Data/mnt/pos'


## Pretrained models

## Experiments
`python trainer.py --task_name=[task_name] --task_mode=[task_mode] --exp_name=[exp_name]`
[task\_name] includes: reg, cls, gan, gan\_cifar10, mnt;
[task\_mode] includes: train, test, baseline;
[exp\_name] can be any string you like.

Example:
Train a controller on regression task:
`python trainer.py --task_name=reg --task_mode=train --exp_name=reg_train`

After the training of the controller, you want to use the controller to guide the training of the regression model on a new dataset:
`python trainer.py --task_name=reg --task_mode=test --exp_name=reg_train`
The program will automaticly load the controller trained in experiment with 'exp\_name' reg_train. So make sure the exp\_name is the same with that in training session. 
Alternatively, you can specify the checkpoint folder where the controller model is saved by using the parameter 'load\_ctrl':
`python trainer.py --task_name=reg --task_mode=test --exp_name=reg_test --load_ctrl=path/to/checkpoint/folder/`

Then you may want to compare the result with a baseline training schedule:
`python trainer.py --task_name=reg --task_mode=baseline --exp_name=reg_baseline`
You can design your own training schedule through the class 'controller_designed' in 'models/reg.py'

## Citation
If you use any part of this code in your research, please cite our paper:
```
@misc{xu2018autoloss,
    title={AutoLoss: Learning Discrete Schedules for Alternate Optimization},
    author={Haowen Xu and Hao Zhang and Zhiting Hu and Xiaodan Liang and Ruslan Salakhutdinov and Eric Xing},
    year={2018},
    eprint={1810.02442},
    archivePrefix={arXiv},
    primaryClass={cs.LG}
}
```
