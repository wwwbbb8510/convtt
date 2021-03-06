import re
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from collections import OrderedDict
from .init import LSUVinit


def custom_densenet(block_config, **kwargs):
    model = DenseNet(num_init_features=16, block_config=block_config, custom_growth_rate=True,
                     **kwargs)
    return model


def densenet40(**kwargs):
    """
    Densenet-40 model
    """
    model = DenseNet(num_init_features=16, growth_rate=12, block_config=(6, 6, 6),
                     **kwargs)
    return model


def densenet100_12(**kwargs):
    """
    Densenet-100-12 model
    """
    model = DenseNet(num_init_features=16, growth_rate=12, block_config=(16, 16, 16),
                     **kwargs)
    return model


def densenet100_24(**kwargs):
    """
    Densenet-100-24 model
    """
    model = DenseNet(num_init_features=16, growth_rate=24, block_config=(16, 16, 16),
                     **kwargs)
    return model


def densenet121(**kwargs):
    """
    Densenet-121 model from
    """
    model = DenseNet(num_init_features=64, growth_rate=32, block_config=(6, 12, 24, 16),
                     **kwargs)
    return model


def densenet169(**kwargs):
    """
    Densenet-169 model from
    """
    model = DenseNet(num_init_features=64, growth_rate=32, block_config=(6, 12, 32, 32),
                     **kwargs)
    return model


def densenet201(**kwargs):
    """
    Densenet-201 model from
    """
    model = DenseNet(num_init_features=64, growth_rate=32, block_config=(6, 12, 48, 32),
                     **kwargs)
    return model


def densenet161(**kwargs):
    """
    Densenet-161 model from
    """
    model = DenseNet(num_init_features=96, growth_rate=48, block_config=(6, 12, 36, 24),
                     **kwargs)
    return model


def init_weights(ms):
    for m in ms.modules():
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            m.weight.data = init.kaiming_normal_(m.weight.data)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)
        elif classname.find('Linear') != -1:
            m.weight.data = init.kaiming_normal_(m.weight.data)


def init_batch_norm(ms):
    for m in ms.modules():
        classname = m.__class__.__name__
        if classname.find('BatchNorm') != -1:
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)


def init_weights_lsuv(model, data):
    LSUVinit(model, data, needed_std=1.0, std_tol=0.1, max_attempts=10, do_orthonorm=True,
             cuda=torch.cuda.is_available())
    model.apply(init_batch_norm)


class _DenseLayer(nn.Sequential):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        super(_DenseLayer, self).__init__()
        self.add_module('norm1', nn.BatchNorm2d(num_input_features)),
        self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module('conv1', nn.Conv2d(num_input_features, bn_size *
                                           growth_rate, kernel_size=1, stride=1, bias=False)),
        self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate)),
        self.add_module('relu2', nn.ReLU(inplace=True)),
        self.add_module('conv2', nn.Conv2d(bn_size * growth_rate, growth_rate,
                                           kernel_size=3, stride=1, padding=1, bias=False)),
        self.drop_rate = drop_rate

    def forward(self, x):
        new_features = super(_DenseLayer, self).forward(x)
        if self.drop_rate > 0 and self.training:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return torch.cat([x, new_features], 1)


class _DenseBlock(nn.Sequential):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(num_input_features + i * growth_rate, growth_rate, bn_size, drop_rate)
            self.add_module('denselayer%d' % (i + 1), layer)


class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features,
                                          kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))


class DenseNet(nn.Module):
    r"""Densenet-BC model class, based on
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_

    Args:
        growth_rate (int) - how many filters to add each layer (`k` in paper)
        block_config (list of 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
          (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
        num_classes (int) - number of classification classes
    """

    def __init__(self, growth_rate=32, block_config=(6, 12, 24, 16),
                 num_init_features=64, bn_size=4, drop_rate=0, num_classes=1000, image_shape=(3, 224, 224),
                 custom_growth_rate=False):
        self.num_connections = 0  # not used
        self.image_shape = image_shape
        self.features = None
        self.avgpool = None
        self.classifier = None

        super(DenseNet, self).__init__()

        if custom_growth_rate:
            self.init_bock_custom(block_config, num_init_features, bn_size, drop_rate, num_classes)
        else:
            if len(block_config) == 3:
                self.init_bock_3(growth_rate, block_config, num_init_features, bn_size, drop_rate, num_classes)
            else:
                self.init_bock_4(growth_rate, block_config, num_init_features, bn_size, drop_rate, num_classes)

        # Official init from torch repo.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def init_bock_custom(self, block_config, num_init_features, bn_size, drop_rate, num_classes):
        width, height = self.image_shape[1], self.image_shape[2]
        # First convolution and pooling
        # generate the parameters of conv1
        conv0_kernel_size, conv0_stride, conv0_padding = self._generate_first_conv_parameters()
        conv0_input_channel = self.image_shape[0]
        has_max_pool_in_features = True if conv0_kernel_size == 7 else False
        features_layers = [
            ('conv0',
             nn.Conv2d(conv0_input_channel, num_init_features, kernel_size=conv0_kernel_size, stride=conv0_stride,
                       padding=conv0_padding, bias=False)),
            ('norm0', nn.BatchNorm2d(num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
        ]
        if has_max_pool_in_features:
            features_layers.append(('pool0', nn.MaxPool2d(kernel_size=3, stride=2,
                                                          padding=1)))
        self.features = nn.Sequential(OrderedDict(features_layers))
        width, height = self._calculate_image_size(width, height, conv0_padding, conv0_kernel_size, conv0_stride)
        width, height = self._calculate_image_size(width, height, 1, 3, 2) if has_max_pool_in_features else (
            width, height)

        # Each denseblock
        num_features = num_init_features
        for i, (num_layers, growth_rate) in enumerate(block_config):
            block = _DenseBlock(num_layers=num_layers, num_input_features=num_features,
                                bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
                width, height = self._calculate_image_size(width, height, 0, 2, 2)
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2

        # Final batch norm
        self.features.add_module('norm5', nn.BatchNorm2d(num_features))

        # Final pooling
        avgpool_kernel_size, avgpool_stride, avgpool_padding = (width, 1, 0)
        self.avgpool = nn.AvgPool2d(avgpool_kernel_size, stride=avgpool_stride, padding=avgpool_padding)
        width, height = self._calculate_image_size(width, height, avgpool_padding, avgpool_kernel_size, avgpool_stride)

        # Linear layer
        self.classifier = nn.Linear(num_features * width * height, num_classes)

    def init_bock_4(self, growth_rate, block_config, num_init_features, bn_size, drop_rate, num_classes):
        width, height = self.image_shape[1], self.image_shape[2]
        # First convolution and pooling
        # generate the parameters of conv1
        conv0_kernel_size, conv0_stride, conv0_padding = self._generate_first_conv_parameters()
        conv0_input_channel = self.image_shape[0]
        self.features = nn.Sequential(OrderedDict([
            ('conv0',
             nn.Conv2d(conv0_input_channel, num_init_features, kernel_size=conv0_kernel_size, stride=conv0_stride,
                       padding=conv0_padding, bias=False)),
            ('norm0', nn.BatchNorm2d(num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
        ]))
        width, height = self._calculate_image_size(width, height, conv0_padding, conv0_kernel_size, conv0_stride)
        width, height = self._calculate_image_size(width, height, 1, 3, 2)

        # Each denseblock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(num_layers=num_layers, num_input_features=num_features,
                                bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
                width, height = self._calculate_image_size(width, height, 0, 2, 2)
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2

        # Final batch norm
        self.features.add_module('norm5', nn.BatchNorm2d(num_features))

        # Final pooling
        avgpool_kernel_size, avgpool_stride, avgpool_padding = (width, 1, 0)
        self.avgpool = nn.AvgPool2d(avgpool_kernel_size, stride=avgpool_stride, padding=avgpool_padding)
        width, height = self._calculate_image_size(width, height, avgpool_padding, avgpool_kernel_size, avgpool_stride)

        # Linear layer
        self.classifier = nn.Linear(num_features * width * height, num_classes)

    def init_bock_3(self, growth_rate, block_config, num_init_features, bn_size, drop_rate, num_classes):
        width, height = self.image_shape[1], self.image_shape[2]
        # First convolution and pooling
        # generate the parameters of conv1
        conv0_kernel_size, conv0_stride, conv0_padding = self._generate_first_conv_parameters()
        conv0_input_channel = self.image_shape[0]
        has_max_pool_in_features = True if conv0_kernel_size == 7 else False
        features_layers = [
            ('conv0',
             nn.Conv2d(conv0_input_channel, num_init_features, kernel_size=conv0_kernel_size, stride=conv0_stride,
                       padding=conv0_padding, bias=False)),
            ('norm0', nn.BatchNorm2d(num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
        ]
        if has_max_pool_in_features:
            features_layers.append(('pool0', nn.MaxPool2d(kernel_size=3, stride=2,
                                                          padding=1)))
        self.features = nn.Sequential(OrderedDict(features_layers))
        width, height = self._calculate_image_size(width, height, conv0_padding, conv0_kernel_size, conv0_stride)
        width, height = self._calculate_image_size(width, height, 1, 3, 2) if has_max_pool_in_features else (
            width, height)

        # Each denseblock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(num_layers=num_layers, num_input_features=num_features,
                                bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
                width, height = self._calculate_image_size(width, height, 0, 2, 2)
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2

        # Final batch norm
        self.features.add_module('norm5', nn.BatchNorm2d(num_features))

        # Final pooling
        avgpool_kernel_size, avgpool_stride, avgpool_padding = (width, 1, 0)
        self.avgpool = nn.AvgPool2d(avgpool_kernel_size, stride=avgpool_stride, padding=avgpool_padding)
        width, height = self._calculate_image_size(width, height, avgpool_padding, avgpool_kernel_size, avgpool_stride)

        # Linear layer
        self.classifier = nn.Linear(num_features * width * height, num_classes)

    def forward(self, x):
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _generate_first_conv_parameters(self):
        """
        generate the parameters of the first conv layer
        :return: kernel_size, stride, padding
        """
        kernel_size, stride, padding = 7, 2, 3
        if self.image_shape[1] < 128:
            kernel_size, stride, padding = 3, 1, 1
        return kernel_size, stride, padding

    def _calculate_image_size(self, width, height, padding_num, kernel_size, stride_size):
        """
        calculate the image size based on the current size, padding, kernel and stride
        :param width:
        :param height:
        :param padding_num:
        :param kernel_size:
        :param stride_size:
        :return: width, height
        """
        height = math.floor((height + padding_num * 2 - kernel_size) / stride_size + 1)
        width = math.floor((width + padding_num * 2 - kernel_size) / stride_size + 1)
        return width, height
