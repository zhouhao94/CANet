# CANet
An implementation of RGB-D co-attention network

## Table of Contents

- [Dependencies](#dependencies)
- [Install](#install)
- [Usage](#usage)

## Dependencies

PyTorch 0.4.0, TensorboardX 1.2 and other packages listed in `requirements.txt`.

## Install

This project uses pytorch and other dependencies, you can intall it with this
```sh
$ pip install -r requirements.txt
```

## Usage

For training, you can pass the following argument,

```
python ACNet_train_V1_nyuv2.py --cuda -b 4
```

For inference, you should run the [ACNet_eval_nyuv2.py](ACNet_eval_nyuv2.py) like this,

```
python ACNet_eval_nyuv2.py --cuda --last-ckpt /path/to/pretrained/model.pth
```
