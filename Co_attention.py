###########################################################################
# Created by: CASIA IVA
# Email: jliu@nlpr.ia.ac.cn
# Copyright (c) 2018
###########################################################################

import numpy as np
import torch
import math
from torch.nn import Module, Sequential, Conv2d, ReLU,AdaptiveMaxPool2d, AdaptiveAvgPool2d, \
    NLLLoss, BCELoss, CrossEntropyLoss, AvgPool2d, MaxPool2d, Parameter, Linear, Sigmoid, Softmax, Dropout, Embedding
from torch.nn import functional as F
from torch.autograd import Variable
torch_ver = torch.__version__[:3]

__all__ = ['PCAM_Module', 'CCAM_Module', 'FusionLayer']


class PCAM_Module(Module):
    """ Position attention module"""
    #Ref from SAGAN
    def __init__(self, in_dim):
        super(PCAM_Module, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.key_conv = Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.value_conv = Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = Parameter(torch.zeros(1))

        self.softmax = Softmax(dim=-1)
    def forward(self, x, y):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X (HxW) X (HxW)
        """
        m_batchsize, C, height, width = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width*height).permute(0, 2, 1)
        proj_key = self.key_conv(y).view(m_batchsize, -1, width*height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(y).view(m_batchsize, -1, width*height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma*out + x
        return out


class CCAM_Module(Module):
    """ Channel attention module"""
    def __init__(self, in_dim):
        super(CCAM_Module, self).__init__()
        self.chanel_in = in_dim


        self.gamma = Parameter(torch.zeros(1))
        self.softmax  = Softmax(dim=-1)
    def forward(self, x, y):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X C X C
        """
        m_batchsize, C, height, width = x.size()
        proj_query = x.view(m_batchsize, C, -1)
        proj_key = y.view(m_batchsize, C, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy)-energy
        attention = self.softmax(energy_new)
        proj_value = y.view(m_batchsize, C, -1)

        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma*out + x
        return out


class FusionLayer(Module):
    def __init__(self, in_channels, groups=1, radix=2, reduction_factor=4, norm_layer=None):
        super(FusionLayer, self).__init__()
        inter_channels = max(in_channels//reduction_factor, 32)
        self.radix = radix
        self.cardinality = groups
        self.use_bn = norm_layer is not None
        self.relu = ReLU(inplace=True)
        self.fc1_p = Conv2d(in_channels, inter_channels, 1, groups=self.cardinality)  # 1024 -> 256
        self.fc1_c = Conv2d(in_channels, inter_channels, 1, groups=self.cardinality)  # 1024 -> 256
        if self.use_bn:
            self.bn1_p = norm_layer(inter_channels)
            self.bn1_c = norm_layer(inter_channels)
        self.fc2_p = Conv2d(inter_channels, in_channels*radix, 1, groups=self.cardinality)  # 256 -> 1024
        self.fc2_c = Conv2d(inter_channels, in_channels*radix, 1, groups=self.cardinality)  # 256 -> 1024

        self.rsoftmax = rSoftMax(radix, groups)

    def forward(self, x, y, z):
        """

        :param x: convolution fusion features
        :param y: position attention features
        :param z: channel attention features
        :return:
        """

        assert self.radix == 2, "Error radix size!"

        batch, rchannel = x.shape[:2] # n, 2048
        if self.radix > 1:
            splited = torch.split(x, rchannel//self.radix, dim=1)
            gap_1 = splited[0]
            gap_2 = splited[1]
        else:
            gap_1 = x
            gap_2 = x

        assert gap_1.shape[1] == y.shape[1], "Error!"
        assert gap_2.shape[1] == z.shape[1], "Error!"

        gap_p = sum([gap_1, y])
        gap_c = sum([gap_2, z])

        gap_p = F.adaptive_avg_pool2d(gap_p, 1)  # n, 512, h, w -> n, 512, 1, 1
        gap_c = F.adaptive_avg_pool2d(gap_c, 1)  # n, 512, h, w -> n, 512, 1, 1

        gap_p = self.fc1_p(gap_p)
        gap_c = self.fc1_c(gap_c)

        if self.use_bn:
            gap_p = self.bn1_p(gap_p)
            gap_c = self.bn1_c(gap_c)

        gap_p = self.relu(gap_p)
        gap_c = self.relu(gap_c)

        atten_p = self.fc2_p(gap_p)  # n, 256, 1, 1 -> n, 1024, 1, 1
        atten_c = self.fc2_c(gap_c)  # n, 256, 1, 1 -> n, 1024, 1, 1

        atten_p = self.rsoftmax(atten_p).view(batch, -1, 1, 1)  # (n, 1024) -> (n, 1024, 1, 1)
        atten_c = self.rsoftmax(atten_c).view(batch, -1, 1, 1)  # (n, 1024) -> (n, 1024, 1, 1)

        if self.radix > 1:
            attens_p = torch.split(atten_p, rchannel//self.radix, dim=1)  # 2(n, 512, 1, 1) tuple
            attens_c = torch.split(atten_c, rchannel//self.radix, dim=1)  # 2(n, 512, 1, 1) tuple

            splited_p = (gap_1, y)  # 2(n, 512, h, w)
            splited_c = (gap_1, y)  # 2(n, 512, h, w)

            out_p = sum([att * split for (att, split) in zip(attens_p, splited_p)])
            out_c = sum([att * split for (att, split) in zip(attens_c, splited_c)])
        else:
            out_p = atten_p * y
            out_c = atten_c * z

        if self.radix > 1:
            out = torch.cat([out_p, out_c], 1)
        else:
            out = sum([out_p, out_c])

        return out.contiguous()


class rSoftMax(Module):
    def __init__(self, radix, cardinality):
        super(rSoftMax, self).__init__()
        self.radix = radix
        self.cardinality = cardinality

    def forward(self, x):
        batch = x.size(0)
        if self.radix > 1:
            x = x.view(batch, self.cardinality, self.radix, -1).transpose(1, 2)  # (n, 1, 2, 512) -> (n, 2, 1, 512)
            x = F.softmax(x, dim=1)
            x = x.reshape(batch, -1)  # (n, 2, 1, 1024) -> (n, 2048)
        else:
            x = torch.sigmoid(x)

        return x
