# CANet
An implementation of RGB-D co-attention network

## Table of Contents

- [Start](#Start)
- [Dependencies](#dependencies)
- [Usage](#usage)

## Start
Clone this repo:
```
git clone https://github.com/zhouhao94/CANet.git
```

## Dependencies
```
Python 3
Pytorch 0.4.1
TensorboardX
Tensorboard
```

install all dependent libraries:
```sh
$ pip install -r requirements.txt
```

## Usage

For training, you can pass the following argument,

```
python CANet_train_nyuv2.py --cuda -b 4
```

For inference, you should run the [CANet_eval_nyuv2.py](CANet_eval_nyuv2.py) like this,

```
python CANet_eval_nyuv2.py --cuda --last-ckpt /path/to/pretrained/model.pth
```
