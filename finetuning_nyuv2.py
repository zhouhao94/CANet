import argparse
import os
import time
import torch

from torch.utils.data import DataLoader
import torch.optim
import torchvision.transforms as transforms
from torchvision.utils import make_grid
from torch import nn

from tensorboardX import SummaryWriter

import CANet_models
import CANet_data_nyuv2 as ACNet_data
from utils import utils
from utils.utils import save_ckpt
from utils.utils import load_ckpt
from utils.utils import print_log
from torch.optim.lr_scheduler import LambdaLR

nyuv2_frq = [0.04636878, 0.10907704, 0.152566  , 0.28470833, 0.29572534,
       0.42489686, 0.49606689, 0.49985867, 0.45401091, 0.52183679,
       0.50204292, 0.74834397, 0.6397011 , 1.00739467, 0.80728748,
       1.01140891, 1.09866549, 1.25703345, 0.9408835 , 1.56565388,
       1.19434108, 0.69079067, 1.86669642, 1.908     , 1.80942453,
       2.72492965, 3.00060817, 2.47616595, 2.44053651, 3.80659652,
       3.31090131, 3.9340523 , 3.53262803, 4.14408881, 3.71099056,
       4.61082739, 4.78020462, 0.44061509, 0.53504894, 0.21667766]

parser = argparse.ArgumentParser(description='RGBD Sementic Segmentation')
parser.add_argument('--data-dir', default=None, metavar='DIR',
                    help='path to SUNRGB-D')
parser.add_argument('--cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 8)')
parser.add_argument('--epochs', default=1000, type=int, metavar='N',
                    help='number of total epochs to run (default: 1500)')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=5, type=int,
                    metavar='N', help='mini-batch size (default: 10)')
parser.add_argument('--lr', '--learning-rate', default=2e-3, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--print-freq', '-p', default=100, type=int,
                    metavar='N', help='print batch frequency (default: 50)')
parser.add_argument('--save-epoch-freq', '-s', default=5, type=int,
                    metavar='N', help='save epoch frequency (default: 5)')
parser.add_argument('--last-ckpt', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--lr-decay-rate', default=0.8, type=float,
                    help='decay rate of learning rate (default: 0.8)')
# hxx
parser.add_argument('--lr-epoch-per-decay', default=5, type=int,
                    help='epoch of per decay of learning rate (default: 150)')
parser.add_argument('--ckpt-dir', default='./model/', metavar='DIR',
                    help='path to save checkpoints')
parser.add_argument('--summary-dir', default='./summary', metavar='DIR',
                    help='path to save summary')
parser.add_argument('--checkpoint', action='store_true', default=False,
                    help='Using Pytorch checkpoint or not')

args = parser.parse_args()
device = torch.device("cuda:0" if args.cuda and torch.cuda.is_available() else "cpu")
image_w = 640
image_h = 480
def train():
    train_data = ACNet_data.SUNRGBD(transform=transforms.Compose([ACNet_data.scaleNorm(),
                                                                   ACNet_data.RandomScale((1.0, 1.4)),
                                                                   ACNet_data.RandomHSV((0.9, 1.1),
                                                                                         (0.9, 1.1),
                                                                                         (25, 25)),
                                                                   ACNet_data.RandomCrop(image_h, image_w),
                                                                   ACNet_data.RandomFlip(),
                                                                   ACNet_data.ToTensor(),
                                                                   ACNet_data.Normalize()]),
                                     phase_train=True,
                                     data_dir=args.data_dir)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=False)

    num_train = len(train_data)

    if args.last_ckpt:
        model = CANet_models.ACNet(num_class=40, pretrained=False)
    else:
        model = CANet_models.ACNet(num_class=40, pretrained=True)
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)
    # CEL_weighted = utils.CrossEntropyLoss2d()
    CEL_weighted = utils.FocalLoss2d(weight=nyuv2_frq, gamma=2)
    model.train()
    model.to(device)
    CEL_weighted.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,
                                momentum=args.momentum, weight_decay=args.weight_decay)

    global_step = 0

    if args.last_ckpt:
        global_step, args.start_epoch = load_ckpt(model, optimizer, args.last_ckpt, device)

    #hxx for finetuing
    # lr_decay_lambda = lambda epoch: 0.2 * args.lr_decay_rate ** ((epoch - args.start_epoch) // args.lr_epoch_per_decay)
    lr_decay_lambda = lambda epoch: (1 - (epoch - args.start_epoch) / (args.epochs - args.start_epoch)) ** 0.9
    scheduler = LambdaLR(optimizer, lr_lambda=lr_decay_lambda)

    writer = SummaryWriter(args.summary_dir)

    for epoch in range(int(args.start_epoch), args.epochs):
        # if (epoch - args.start_epoch) % args.lr_epoch_per_decay == 0:
        scheduler.step(epoch)
        local_count = 0
        last_count = 0
        end_time = time.time()
        if epoch % args.save_epoch_freq == 0 and epoch != args.start_epoch:
            save_ckpt(args.ckpt_dir, model, optimizer, global_step, epoch,
                      local_count, num_train)

        for batch_idx, sample in enumerate(train_loader):

            image = sample['image'].to(device)
            depth = sample['depth'].to(device)
            target_scales = [sample[s].to(device) for s in ['label', 'label2', 'label3', 'label4', 'label5']]
            optimizer.zero_grad()
            pred_scales = model(image, depth, args.checkpoint)
            loss = CEL_weighted(pred_scales, target_scales)
            loss.backward()
            optimizer.step()
            local_count += image.data.shape[0]
            global_step += 1
            if global_step % args.print_freq == 0 or global_step == 1:

                time_inter = time.time() - end_time
                count_inter = local_count - last_count
                print_log(global_step, epoch, local_count, count_inter,
                          num_train, loss, time_inter)
                end_time = time.time()

                last_count = local_count

    save_ckpt(args.ckpt_dir, model, optimizer, global_step, args.epochs,
              0, num_train)

    print("Training completed ")

if __name__ == '__main__':
    if not os.path.exists(args.ckpt_dir):
        os.mkdir(args.ckpt_dir)
    if not os.path.exists(args.summary_dir):
        os.mkdir(args.summary_dir)

    train()
