# coding=utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init


# Basic Convolution Block
class BasicConv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride=1, padding=0,
                 dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_ch
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_ch, momentum=0.01) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)

        return x


class MLP(nn.Module):
    """
    Multilayer perception block
    :param
    channels: int
        number of input/output channels
    reduction_ratio: int, default=16
        channel reduction ratio
    """
    def __init__(self, channels, reduction_ratio=16):
        super(MLP, self).__init__()
        mid_channels = channels // reduction_ratio
        self.fc1 = nn.Linear(channels, mid_channels)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(mid_channels, channels)

    def forward(self, x):
        x = x.view(x.size(0), -1)        # flatten
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)

        return x


class channelGate(nn.Module):
    def __init__(self, gate_ch, reduction_ratio=16):
        super(channelGate, self).__init__()
        self.mlp = MLP(channels=gate_ch, reduction_ratio=reduction_ratio)

    def forward(self, x):
        # global average pooling
        att1 = F.avg_pool2d(x, kernel_size=x.shape[2:], stride=x.shape[2:])
        att1 = self.mlp(att1)
        # max pooling
        att2 = F.max_pool2d(x, kernel_size=x.shape[2:], stride=x.shape[2:])
        att2 = self.mlp(att2)
        att = att1 + att2
        scale = F.sigmoid(att).unsqueeze(2).unsqueeze(3).expand_as(x)

        return x*scale


class spatialGate(nn.Module):
    def __init__(self):
        super(spatialGate, self).__init__()
        self.spatial = BasicConv(in_ch=2, out_ch=1, kernel_size=7, stride=1, padding=3, relu=False)

    def forward(self, x):
        # max pooling
        att1 = torch.max(x, 1)[0].unsqueeze(1)
        att2 = torch.mean(x, 1).unsqueeze(1)
        att = torch.cat((att1, att2), dim=1)
        att = self.spatial(att)
        scale = F.sigmoid(att).expand_as(x)

        return x*scale


# Convolutional Block Attention Module
class cbamBlock(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16):
        super(cbamBlock, self).__init__()
        self.channelGate = channelGate(gate_ch=gate_channels, reduction_ratio=reduction_ratio)   # channel attention
        self.spatialGate = spatialGate()

    def forward(self, x):
        out = self.channelGate(x)
        out = self.spatialGate(out)

        return out


# Resnet Module
def conv3x3(inplanes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(inplanes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, use_cbam=False):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

        if use_cbam:
            self.cbam = cbamBlock(planes, reduction_ratio=16)
        else:
            self.cbam = None

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        if self.cbam is not None:
            out = self.cbam(out)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, use_cbam=False):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

        if use_cbam:
            self.cbam = cbamBlock(planes*4, reduction_ratio=16)
        else:
            self.cbam = None

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

        if self.cbam is not None:
            out = self.cbam(out)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000):
        super(ResNet, self).__init__()
        self.inplanes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(kernel_size=7, stride=1)
        self.fc = nn.Linear(512*block.expansion, num_classes)

        init.kaiming_normal_(self.fc.weight)     # Kaming initial
        for key in self.state_dict():
            if key.split('.')[-1] == "weight":
                if "conv" in key:
                    init.kaiming_normal_(self.state_dict()[key], mode='fan_out')
                if "bn" in key:
                    if "spatialGate" in key:
                        self.state_dict()[key][...] = 0
                    else:
                        self.state_dict()[key][...] = 1
            elif key.split('.')[-1] == "bias":
                self.state_dict()[key][...] = 0

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, use_cbam=True))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, use_cbam=True))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)     # flatten
        x = self.fc(x)

        return x


def res_cbam(depth):

    if depth == 18:
        model = ResNet(BasicBlock, layers=[2, 2, 2, 2])

    elif depth == 34:
        model = ResNet(BasicBlock, layers=[3, 4, 6, 3])

    elif depth == 50:
        model = ResNet(Bottleneck, layers=[3, 4, 6, 3])

    elif depth == 101:
        model = ResNet(Bottleneck, layers=[3, 4, 23, 3])

    elif depth == 152:
        model = ResNet(Bottleneck, layers=[3, 8, 36, 3])

    return model





