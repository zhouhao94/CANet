# CANet
CANet: Co-attention network for RGB-D semantic segmentation

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
PyTorch 0.4.0, TensorboardX and other packages listed in `requirements.txt`:
```
torch==0.4.1
torchvision=0.2.2
numpy
imageio
scipy
tensorboardX
matplotlib
scikit-image
h5py
```

Install all dependent libraries:
```sh
$ pip install -r requirements.txt
```

## Usage

For training, you can pass the following argument:

```
python CANet_train_nyuv2.py --cuda -b 4
```

For inference, you should run the [CANet_eval_nyuv2.py](CANet_eval_nyuv2.py) like this:

```
python CANet_eval_nyuv2.py --cuda --last-ckpt /path/to/pretrained/model.pth
```
