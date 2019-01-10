AutoLoss: Learning Discrete Schedule for Alternate Optimization
======================
Code for reproducing experiments in ["AutoLoss: Learning Discrete Schedule for Alternate Optimization"] (https://arxiv.org/abs/1810.02442).

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
