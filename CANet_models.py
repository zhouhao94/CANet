import torch
from torch import nn
from torch.nn import functional as F
import math
import torch.utils.model_zoo as model_zoo
from utils import utils
from torch.utils.checkpoint import checkpoint
from Co_attention import PCAM_Module, CCAM_Module, FusionLayer
model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

class ACNet(nn.Module):
    def __init__(self, num_class=37, backbone='ResNet-101', pretrained=False, pcca5=False):
        super(ACNet, self).__init__()

        self.pcca5 = pcca5
        self.backbone = backbone

        if self.backbone == 'ResNet-50':
            layers = [3, 4, 6, 3]
        else:
            layers = [3, 4, 23, 3]

        block = Bottleneck
        transblock = TransBasicBlock
        # RGB image branch
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2) # use PSPNet extractors
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        # depth image branch
        self.inplanes = 64
        self.conv1_d = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1_d = nn.BatchNorm2d(64)
        self.relu_d = nn.ReLU(inplace=True)
        self.maxpool_d = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1_d = self._make_layer(block, 64, layers[0])
        self.layer2_d = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3_d = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4_d = self._make_layer(block, 512, layers[3], stride=2)

        """
        # merge branch
        self.atten_rgb_0 = self.channel_attention(64)
        self.atten_depth_0 = self.channel_attention(64)
        self.maxpool_m = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.atten_rgb_1 = self.channel_attention(64*4)
        self.atten_depth_1 = self.channel_attention(64*4)
        # self.conv_2 = nn.Conv2d(64*4, 64*4, kernel_size=1) #todo 用cat和conv降回通道数
        self.atten_rgb_2 = self.channel_attention(128*4)
        self.atten_depth_2 = self.channel_attention(128*4)
        self.atten_rgb_3 = self.channel_attention(256*4)
        self.atten_depth_3 = self.channel_attention(256*4)
        self.atten_rgb_4 = self.channel_attention(512*4)
        self.atten_depth_4 = self.channel_attention(512*4)
        """

        self.inplanes = 64
        self.layer1_m = self._make_layer(block, 64, layers[0])
        self.layer2_m = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3_m = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4_m = self._make_layer(block, 512, layers[3], stride=2)

        # agant module
        self.agant0 = self._make_agant_layer(64, 64)
        self.agant1 = self._make_agant_layer(64*4, 64)
        self.agant2 = self._make_agant_layer(128*4, 128)
        self.agant3 = self._make_agant_layer(256*4, 256)
        self.agant4 = self._make_agant_layer(512*4, 512)

        #transpose layer
        self.inplanes = 512
        self.deconv1 = self._make_transpose(transblock, 256, 6, stride=2)
        self.deconv2 = self._make_transpose(transblock, 128, 4, stride=2)
        self.deconv3 = self._make_transpose(transblock, 64, 3, stride=2)
        self.deconv4 = self._make_transpose(transblock, 64, 3, stride=2)

        # final blcok
        self.inplanes = 64
        self.final_conv = self._make_transpose(transblock, 64, 3)

        self.final_deconv = nn.ConvTranspose2d(self.inplanes, num_class, kernel_size=2,
                                               stride=2, padding=0, bias=True)

        self.out5_conv = nn.Conv2d(256, num_class, kernel_size=1, stride=1, bias=True)
        self.out4_conv = nn.Conv2d(128, num_class, kernel_size=1, stride=1, bias=True)
        self.out3_conv = nn.Conv2d(64, num_class, kernel_size=1, stride=1, bias=True)
        self.out2_conv = nn.Conv2d(64, num_class, kernel_size=1, stride=1, bias=True)

        if self.pcca5:

            self.conv_5a = nn.Sequential(nn.Conv2d(2048, 512, kernel_size=3, stride=1, padding=1, bias=False),
                                     nn.BatchNorm2d(512),
                                     nn.ReLU())
            self.conv_5c = nn.Sequential(nn.Conv2d(2048, 512, kernel_size=3, stride=1, padding=1, bias=False),
                                     nn.BatchNorm2d(512),
                                     nn.ReLU())
            self.pca_5 = PCAM_Module(512)
            self.cca_5 = CCAM_Module(512)
            """
            self.pconv_5 = nn.Sequential(nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False),
                                     BatchNorm2d(512),
                                     nn.ReLU())
            self.cconv_5 = nn.Sequential(nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False),
                                     BatchNorm2d(512),
                                     nn.ReLU())
            self.pconv_out = nn.Sequential(nn.Conv2d(512, 2048, kernel_size=3, stride=1, padding=1, bias=False),
                                     BatchNorm2d(2048),
                                     nn.ReLU(),
                                     nn.Dropout2d(0.1, False))
            self.cconv_out = nn.Sequential(nn.Conv2d(512, 2048, kernel_size=3, stride=1, padding=1, bias=False),
                                     BatchNorm2d(2048),
                                     nn.ReLU(),
                                     nn.Dropout2d(0.1, False))
            self.alpha = Parameter(torch.ones(1))
            self.beta = Parameter(torch.ones(1))
            """
            self.pconv_5 = nn.Sequential(nn.Conv2d(512, 1024, kernel_size=1, stride=1, padding=0, bias=False),
                                     nn.BatchNorm2d(1024),
                                     nn.ReLU())
            self.cconv_5 = nn.Sequential(nn.Conv2d(512, 1024, kernel_size=1, stride=1, padding=0, bias=False),
                                     nn.BatchNorm2d(1024),
                                     nn.ReLU())
            self.split_conv = FusionLayer(in_channels=1024, groups=1,radix=2, reduction_factor=4, norm_layer=nn.BatchNorm2d)

        # weight initial
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        if pretrained:
            self._load_resnet_pretrained()

    def encoder(self, rgb, depth):
        rgb = self.conv1(rgb)
        rgb = self.bn1(rgb)
        rgb = self.relu(rgb)
        depth = self.conv1_d(depth)
        depth = self.bn1_d(depth)
        depth = self.relu_d(depth)

        m0 = rgb + depth

        rgb = self.maxpool(rgb)
        depth = self.maxpool_d(depth)
        m = self.maxpool(m0)

        # block 1
        rgb = self.layer1(rgb)
        depth = self.layer1_d(depth)
        m = self.layer1_m(m)

        m1 = m + rgb + depth

        # block 2
        rgb = self.layer2(rgb)
        depth = self.layer2_d(depth)
        m = self.layer2_m(m1)

        m2 = m + rgb + depth

        # block 3
        rgb = self.layer3(rgb)
        depth = self.layer3_d(depth)
        m = self.layer3_m(m2)

        m3 = m + rgb + depth

        # block 4
        rgb = self.layer4(rgb)
        depth = self.layer4_d(depth)
        m = self.layer4_m(m3)

        if self.pcca5:
            rgb_down = self.conv_5a(rgb)
            depth_down = self.conv_5c(depth)
            attention_position = self.pca_5(rgb_down, depth_down)
            attention_channel = self.cca_5(rgb_down, depth_down)
            p_out = self.pconv_5(attention_position)
            c_out = self.cconv_5(attention_channel)
            m4 = self.split_conv(m, p_out, c_out)

            """
            smooth_p = self.pconv_5(attention_position)
            smooth_c = self.cconv_5(attention_channel)
            p_out = self.pconv_out(smooth_p)
            c_out = self.cconv_out(smooth_c)
            m4 = m + self.alpha * p_out + self.beta * c_out
            """
        else:
            m4 = m + rgb + depth

        return m0, m1, m2, m3, m4  # channel of m is 2048

    def decoder(self, fuse0, fuse1, fuse2, fuse3, fuse4):
        agant4 = self.agant4(fuse4)
        # upsample 1
        x = self.deconv1(agant4)
        if self.training:
            out5 = self.out5_conv(x)
        x = x + self.agant3(fuse3)
        # upsample 2
        x = self.deconv2(x)
        if self.training:
            out4 = self.out4_conv(x)
        x = x + self.agant2(fuse2)
        # upsample 3
        x = self.deconv3(x)
        if self.training:
            out3 = self.out3_conv(x)
        x = x + self.agant1(fuse1)
        # upsample 4
        x = self.deconv4(x)
        if self.training:
            out2 = self.out2_conv(x)
        x = x + self.agant0(fuse0)
        # final
        x = self.final_conv(x)
        out = self.final_deconv(x)

        if self.training:
            return out, out2, out3, out4, out5

        return out

    def forward(self, rgb, depth, phase_checkpoint=False):
        fuses = self.encoder(rgb, depth)
        m = self.decoder(*fuses)
        return m

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation))

        return nn.Sequential(*layers)

    def channel_attention(self, num_channel, ablation=False):
        # todo add convolution here
        pool = nn.AdaptiveAvgPool2d(1)
        conv = nn.Conv2d(num_channel, num_channel, kernel_size=1)
        # bn = nn.BatchNorm2d(num_channel)
        activation = nn.Sigmoid() # todo modify the activation function

        return nn.Sequential(*[pool, conv, activation])

    def _make_agant_layer(self, inplanes, planes):
        layers = nn.Sequential(
            nn.Conv2d(inplanes, planes, kernel_size=1,
                      stride=1, padding=0, bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True)
        )
        return layers

    def _make_transpose(self, block, planes, blocks, stride=1):
        upsample = None
        if stride != 1:
            upsample = nn.Sequential(
                nn.ConvTranspose2d(self.inplanes, planes,
                                   kernel_size=2, stride=stride,
                                   padding=0, bias=False),
                nn.BatchNorm2d(planes),
            )
        elif self.inplanes != planes:
            upsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )

        layers = []

        for i in range(1, blocks):
            layers.append(block(self.inplanes, self.inplanes))

        layers.append(block(self.inplanes, planes, stride, upsample))
        self.inplanes = planes

        return nn.Sequential(*layers)


    def _load_resnet_pretrained(self):
        if self.backbone == 'ResNet-50':
            pretrain_dict = model_zoo.load_url(utils.model_urls['resnet50'])
        elif self.backbone == 'ResNet-101':
            pretrain_dict = model_zoo.load_url(utils.model_urls['resnet101'])
        model_dict = {}
        state_dict = self.state_dict()
        for k, v in pretrain_dict.items():
            # print('%%%%% ', k)
            if k in state_dict:
                if k.startswith('conv1'):
                    model_dict[k] = v
                    # print('##### ', k)
                    model_dict[k.replace('conv1', 'conv1_d')] = torch.mean(v, 1).data. \
                        view_as(state_dict[k.replace('conv1', 'conv1_d')])

                elif k.startswith('bn1'):
                    model_dict[k] = v
                    model_dict[k.replace('bn1', 'bn1_d')] = v
                elif k.startswith('layer'):
                    model_dict[k] = v
                    model_dict[k[:6]+'_d'+k[6:]] = v
                    model_dict[k[:6]+'_m'+k[6:]] = v
        state_dict.update(model_dict)
        self.load_state_dict(state_dict)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, dilation=dilation,
                               padding=dilation, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class PSPModule(nn.Module):
    def __init__(self, features, out_features=1024, sizes=(1, 2, 3, 6)):
        super().__init__()
        self.stages = []
        self.stages = nn.ModuleList([self._make_stage(features, size) for size in sizes])
        self.bottleneck = nn.Conv2d(features * (len(sizes) + 1), out_features, kernel_size=1)
        self.relu = nn.ReLU()

    def _make_stage(self, features, size):
        prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        conv = nn.Conv2d(features, features, kernel_size=1, bias=False)
        return nn.Sequential(prior, conv)

    def forward(self, feats):
        h, w = feats.size(2), feats.size(3)
        priors = [F.upsample(input=stage(feats), size=(h, w), mode='bilinear') for stage in self.stages] + [feats]
        bottle = self.bottleneck(torch.cat(priors, 1))
        return self.relu(bottle)

class PSPUpsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.PReLU()
        )

    def forward(self, x):
        h, w = 2 * x.size(2), 2 * x.size(3)
        p = F.upsample(input=x, size=(h, w), mode='bilinear')
        return self.conv(p)


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class TransBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, upsample=None, **kwargs):
        super(TransBasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, inplanes)
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.relu = nn.ReLU(inplace=True)
        if upsample is not None and stride != 1:
            self.conv2 = nn.ConvTranspose2d(inplanes, planes,
                                            kernel_size=3, stride=stride, padding=1,
                                            output_padding=1, bias=False)
        else:
            self.conv2 = conv3x3(inplanes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.upsample = upsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.upsample is not None:
            residual = self.upsample(x)

        out += residual
        out = self.relu(out)

        return out
