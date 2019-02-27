AutoLoss: Learning Discrete Schedules for Alternate Optimization
======================
Code for reproducing experiments in [AutoLoss: Learning Discrete Schedules for Alternate Optimization](https://arxiv.org/abs/1810.02442).

## Requirements
```
python>=3.6, tensorflow-gpu==1.8.0, matplotlib==2.2.2, imageio==2.4.1, sklearn==0.20.2, nltk==3.4, subword-nmt==0.3.6
```

## Getting Started

Install and Activate a Virtual Environment
```
virtualenv --python=python3.6 ./env
source ./env/bin/activate
```
Install Dependencies
```
pip install -r requirements.txt
```

## Datasets
### Regression and Classification

The regression and classification tasks use synthetic data. To synthesize a dataset for the regression or the classification experiment, run the following script:

```
# regression
python dataio/gen_reg_data.py

# classification
python dataio/gen_cls_data.py
```

### GANs
Our GANs experiments use [MNIST](http://yann.lecun.com/exdb/mnist/) and [CIFAR-10](http://www.cs.toronto.edu/~kriz/cifar.html) datasets.
Set a desired path for the argument `data_dir` in config file `config/gan/cfg.py`. 
The training scripts will automatically download the data at its first run.


The Cifar10 database is available at [Download page of CIFAR10]()
You need to download the python version from this page and unzip it. Set the argument 'data\_dir' in config file 'config/gan_cifar.py' to where you save the data.

### Multi-task Neural Translation
You can directly use the preprocessed data we have provided in this repository under the path `data/nmt/`.
Or you can prepare the data by running the following script:
```
python dataio/gen_nmt_data.py
```
In this case, you need to download the `tigercorpus-2.2.xml.tar.gz` from [the TIGER corpus](http://www.ims.uni-stuttgart.de/forschung/ressourcen/korpora/tiger.en.html) and uncompress it to the path `data/mnt/pos`.
The script will expect an XML file named `tiger\_release\_aug07.corrected.16012013.xml`.

*Caution*: The data preparation for the multi-task neural machine translation task could take 10-15 mins on a desktop with Intel i7-6800K CPU @ 3.40GHz x 12 CPU. 


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

## Pretrained models
It may take days to train a controller on GANs task. We provide a pretrained controller for MNIST GAN. To test this controller:
```
python trainer.py --task_name=gan --task_mode=test --exp_name=[any-experiment-name] --load_ctrl=/path/to/AutoLossRelease/weights/gan_2l_adam_short_ctrl
```

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
