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
You just need to set a path to the argument 'data\_dir' in config file '/config/gan/cfg.py'. The database will be downloaded to that folder at the first run.

Similarly, (TODO: for Cifar10)

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
To train a controller for regression task, run:
`python trainer.py --task_name=reg --task_mode=train --exp_name=train_reg_controller`


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
