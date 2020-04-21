import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import numpy as np
import os
from torch.nn import init

DEFAULT_GROWTH_RATE_CONFIG = [[8, 8, 8], [8, 8, 8]]


def densenet_dynamic_gr(growth_rate_config=DEFAULT_GROWTH_RATE_CONFIG,
                        num_init_features=32, bn_size=4, drop_rate=0, num_classes=10, image_shape=(1, 28, 28)):
    """
    initialise
    :param growth_rate_config: configuration of the growth rates. 2-d list
    :type growth_rate_config: list
    :param num_init_features: number of input features
    :type num_init_features: int
    :param bn_size: batch norm size
    :type bn_size: int
    :param drop_rate: dropout rate
    :type drop_rate: float
    :param num_classes: number of classes
    :type num_classes: int
    :param image_shape: image shape
    :type image_shape: tuple
    """
    net = DenseNetDynamicGR(growth_rate_config,
                            bn_size, num_classes, image_shape, num_init_features, drop_rate)
    net.apply(init_weights)
    return net


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


class DenseNetDynamicGR(nn.Module):
    """
    Dynamic Net
    """

    def __init__(self, growth_rate_config,
                 bn_size, num_classes, image_shape=(1, 28, 28), num_init_features=32, drop_rate=0):
        """
        initialise
        :param growth_rate_config: configuration of the growth rates. 2-d list
        :type growth_rate_config: list
        :param num_init_features: number of input features
        :type num_init_features: int
        :param bn_size: batch norm size
        :type bn_size: int
        :param drop_rate: dropout rate
        :type drop_rate: float
        :param num_classes: number of classes
        :type num_classes: int
        :param image_shape: image shape
        :type image_shape: tuple
        """
        self._growth_rate_config = growth_rate_config
        self._num_init_features = num_init_features
        self._bn_size = bn_size
        self._drop_rate = drop_rate
        self._num_classes = num_classes
        self._image_shape = image_shape

        self._num_connections = 0  # not used
        self._features = None
        self._avgpool = None
        self._classifier = None

        super(DenseNetDynamicGR, self).__init__()

        self.init_bocks()

    def init_bocks(self):
        width, height = self.image_shape[1], self.image_shape[2]
        # First convolution and pooling
        # generate the parameters of conv1
        conv0_kernel_size, conv0_stride, conv0_padding = self._generate_first_conv_parameters()
        conv0_input_channel = self.image_shape[0]
        self._features = nn.Sequential(OrderedDict([
            ('conv0',
             nn.Conv2d(conv0_input_channel, self.num_init_features, kernel_size=conv0_kernel_size, stride=conv0_stride,
                       padding=conv0_padding, bias=False)),
            ('norm0', nn.BatchNorm2d(self.num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
            # ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
        ]))
        width, height = self._calculate_image_size(width, height, conv0_padding, conv0_kernel_size, conv0_stride)
        # width, height = self._calculate_image_size(width, height, 1, 3, 2)

        # Each denseblock_dynamic_gr
        num_features = self.num_init_features
        for i, block_config in enumerate(self.growth_rate_config):
            block = _DenseBlockDynamicGR(num_input_features=num_features,
                                         bn_size=self.bn_size, growth_rate=block_config,
                                         drop_rate=self.drop_rate)
            self._features.add_module('denseblock_dynamic_gr%d' % (i + 1), block)
            num_features = block.num_output_features
            if i != len(self.growth_rate_config) - 1:
                trans = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
                width, height = self._calculate_image_size(width, height, 0, 2, 2)
                self._features.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2

        # Final batch norm
        self._features.add_module('norm5', nn.BatchNorm2d(num_features))

        # Final pooling - global average pooling
        avgpool_kernel_size, avgpool_stride, avgpool_padding = (width, 1, 0)
        self._avgpool = nn.AvgPool2d(avgpool_kernel_size, stride=avgpool_stride, padding=avgpool_padding)
        width, height = self._calculate_image_size(width, height, avgpool_padding, avgpool_kernel_size, avgpool_stride)

        # Linear layer
        self._classifier = nn.Linear(num_features * width * height, self.num_classes)

    def forward(self, x):
        features = self._features(x)
        out = F.relu(features, inplace=True)
        out = self._avgpool(out)
        out = out.view(out.size(0), -1)
        out = self._classifier(out)
        return out

    def _generate_first_conv_parameters(self):
        """
        generate the parameters of the first conv layer
        :return: kernel_size, stride, padding
        """
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

    @property
    def growth_rate_config(self):
        return self._growth_rate_config

    @property
    def num_init_features(self):
        return self._num_init_features

    @property
    def bn_size(self):
        return self._bn_size

    @property
    def drop_rate(self):
        return self._drop_rate

    @property
    def num_classes(self):
        return self._num_classes

    @property
    def image_shape(self):
        return self._image_shape


class _DenseLayerDynamicGR(nn.Sequential):
    """
    dynamic layer
    """

    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        """
        initialise
        :param num_input_features: number of input features
        :type num_input_features: int
        :param bn_size: batch norm size
        :type bn_size: int
        :param growth_rate: growth rate of the dynamic layer
        :type growth_rate: int
        :param drop_rate: dropout rate
        :type drop_rate: float
        """
        self._num_input_features = num_input_features
        self._growth_rate = growth_rate
        self._bn_size = bn_size
        self._drop_rate = drop_rate

        super(_DenseLayerDynamicGR, self).__init__()
        self.add_module('norm1', nn.BatchNorm2d(num_input_features)),
        self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module('conv1', nn.Conv2d(num_input_features, bn_size *
                                           growth_rate, kernel_size=1, stride=1, bias=False)),
        self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate)),
        self.add_module('relu2', nn.ReLU(inplace=True)),
        self.add_module('conv2', nn.Conv2d(bn_size * growth_rate, growth_rate,
                                           kernel_size=3, stride=1, padding=1, bias=False)),

    def forward(self, x):
        new_features = super(_DenseLayerDynamicGR, self).forward(x)
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return torch.cat([x, new_features], 1)

    @property
    def num_input_features(self):
        return self._num_input_features

    @property
    def growth_rate(self):
        return self._growth_rate

    @property
    def bn_size(self):
        return self._bn_size

    @property
    def drop_rate(self):
        return self._drop_rate


class _DenseBlockDynamicGR(nn.Sequential):
    """
    dynamic block
    """

    def __init__(self, num_input_features, bn_size, growth_rate, drop_rate):
        """
        initialise
        :param num_input_features: number of input features
        :type num_input_features: int
        :param bn_size: batch norm size
        :type bn_size: int
        :param growth_rate: config of the growth rates of all dynamic layers
        :type growth_rate: list
        :param drop_rate: dropout rate
        :type drop_rate: float
        """
        super(_DenseBlockDynamicGR, self).__init__()

        self._num_layers = len(growth_rate)
        self._arr_outputs = []
        self._num_output_features = None

        # create layers
        self._arr_num_new_features = np.array(
            [num_input_features if i == 0 else growth_rate[i - 1] for i in range(self.num_layers + 1)])
        for i in range(self.num_layers):
            # input index is equal to output index minus 1
            curr_num_input_features = self._arr_num_new_features[0:i + 1].sum().item()
            layer = _DenseLayerDynamicGR(curr_num_input_features, growth_rate[i], bn_size, drop_rate)
            self.add_module('dense_layer_dynamic_gr_%d' % (i + 1), layer)

        # calculate the output features of last layer
        num_connections = self.dict_backward_connections_by_layer['dense_layer_dynamic_gr_%d' % self.num_layers]
        curr_num_input_features = self._arr_num_new_features[num_connections].sum() + growth_rate
        self._num_output_features = curr_num_input_features.item()

    def __repr__(self):
        str_repr = super(_DenseBlockDynamicGR, self).__repr__() + os.linesep
        str_connection = self.dict_connections_by_layer['dense_layer_dynamic_gr_0']
        str_repr += 'input connection: {}'.format(str_connection)
        str_repr += os.linesep
        for i in range(self.num_layers):
            curr_layer_index = i + 1
            str_connection = self.dict_connections_by_layer['dense_layer_dynamic_gr_%d' % curr_layer_index]
            str_repr += 'dynamic layer - {}'.format(curr_layer_index)
            str_repr += os.linesep
            str_repr += 'layer connection: {}'.format(str_connection)
            str_repr += os.linesep

        return str_repr

    @property
    def num_layers(self):
        return self._num_layers

    @property
    def num_output_features(self):
        return self._num_output_features


class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features,
                                          kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))
