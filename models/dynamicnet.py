import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import numpy as np
import os

DEFAULT_GROWTH_RATE_CONFIG = (16, 16)
DEFAULT_LAYER_CONFIG = (4, 4)
DEFAULT_CONNECTION_CONFIG = (
    [[1, 0, 1, 1], [1, 0, 1], [1, 0], [1]],
    [[1, 0, 1, 1], [1, 0, 1], [1, 0], [1]]
)


def dynamicnet(growth_rate_config=DEFAULT_GROWTH_RATE_CONFIG, layer_config=DEFAULT_LAYER_CONFIG,
               connection_config=DEFAULT_CONNECTION_CONFIG,
               num_init_features=32, bn_size=4, drop_rate=0, num_classes=10, image_shape=(1, 28, 28)):
    """
    initialise
    :param growth_rate_config: configuration of the growth rates of dynamic blocks
    :type growth_rate_config: tuple
    :param layer_config: configuration of number of layers of dynamic blocks
    :type layer_config: tuple
    :param connection_config: configurations of connections of dynamic blocks
    :type connection_config: tuple
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
    net = DynamicNet(growth_rate_config, layer_config, connection_config,
                     bn_size, num_classes, image_shape, num_init_features, drop_rate)
    return net


class DynamicNet(nn.Module):
    """
    Dynamic Net
    """

    def __init__(self, growth_rate_config, layer_config, connection_config,
                 bn_size, num_classes, image_shape=(1, 28, 28), num_init_features=32, drop_rate=0):
        """
        initialise
        :param growth_rate_config: configuration of the growth rates of dynamic blocks
        :type growth_rate_config: tuple
        :param layer_config: configuration of number of layers of dynamic blocks
        :type layer_config: tuple
        :param connection_config: configurations of connections of dynamic blocks
        :type connection_config: tuple
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
        self._layer_config = layer_config
        self._connection_config = connection_config
        self._num_init_features = num_init_features
        self._bn_size = bn_size
        self._drop_rate = drop_rate
        self._num_classes = num_classes
        self._image_shape = image_shape

        self._num_connections = 0  # not used
        self._features = None
        self._avgpool = None
        self._classifier = None

        super(DynamicNet, self).__init__()

        self.init_bocks()

        # Official init from torch repo.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant(m.weight, 1)
                nn.init.constant(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant(m.bias, 0)

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

        # Each dynamicblock
        num_features = self.num_init_features
        for i, num_layers in enumerate(self.layer_config):
            block = _DynamicBlock(num_layers=num_layers, num_input_features=num_features,
                                  bn_size=self.bn_size, growth_rate=self.growth_rate_config[i],
                                  connections=self.connection_config[i], drop_rate=self.drop_rate)
            self._features.add_module('dynamicblock%d' % (i + 1), block)
            num_features = block.num_output_features
            if i != len(self.layer_config) - 1:
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
    def layer_config(self):
        return self._layer_config

    @property
    def connection_config(self):
        return self._connection_config

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


class _DynamicLayer(nn.Sequential):
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

        super(_DynamicLayer, self).__init__()
        self.add_module('norm1', nn.BatchNorm2d(num_input_features)),
        self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module('conv1', nn.Conv2d(num_input_features, bn_size *
                                           growth_rate, kernel_size=1, stride=1, bias=False)),
        self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate)),
        self.add_module('relu2', nn.ReLU(inplace=True)),
        self.add_module('conv2', nn.Conv2d(bn_size * growth_rate, growth_rate,
                                           kernel_size=3, stride=1, padding=1, bias=False)),

    def forward(self, x):
        new_features = super(_DynamicLayer, self).forward(x)
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)

        return new_features

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


class _DynamicBlock(nn.Sequential):
    """
    dynamic block
    """

    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, connections, drop_rate):
        """
        initialise
        :param num_layers: number of layers
        :type num_layers: int
        :param num_input_features: number of input features
        :type num_input_features: int
        :param bn_size: batch norm size
        :type bn_size: int
        :param growth_rate: growth rate of the dynamic layer
        :type growth_rate: int
        :param connections: connection topology of dynamic block
        :type connections: list
        :param drop_rate: dropout rate
        :type drop_rate: float
        """
        super(_DynamicBlock, self).__init__()
        self._connections = connections
        # dict to store the connections for each layer, use output index as the layer index starting from 0
        # input as the output of layer 0 and input of layer 1
        # the dictionary store all of the output layers that the current layer will be concatenated to
        self._dict_connections_by_layer = {}
        # dict to store the backward connections for each layer, use output index as the layer index starting from 1
        # the dictionary store all of the output layers that will be concatenated to the current layer
        self._dict_backward_connections_by_layer = {}
        self._num_layers = num_layers
        self._arr_outputs = []
        self._num_output_features = None

        # parse binary string to forward and backward connection dictionary
        self._parse_binary_connections()

        # create layers
        self._arr_num_new_features = np.array(
            [num_input_features if i == 0 else growth_rate for i in range(self.num_layers + 1)])
        for i in range(self.num_layers):
            # input index is equal to output index minus 1
            num_connections = self.dict_backward_connections_by_layer['dynamic_layer_%d' % i]
            curr_num_input_features = self._arr_num_new_features[0] if i == 0 else \
                self._arr_num_new_features[num_connections].sum() + growth_rate
            curr_num_input_features = curr_num_input_features.item()
            layer = _DynamicLayer(curr_num_input_features, growth_rate, bn_size, drop_rate)
            self.add_module('dynamic_layer_%d' % (i + 1), layer)

        # calculate the output features of last layer
        num_connections = self.dict_backward_connections_by_layer['dynamic_layer_%d' % self.num_layers]
        curr_num_input_features = self._arr_num_new_features[num_connections].sum() + growth_rate
        self._num_output_features = curr_num_input_features.item()

    def forward(self, x):
        arr_new_features = [x]
        # layer index from 1
        curr_output = x
        for i in range(self.num_layers):
            curr_layer_index = i + 1
            layer = getattr(self, 'dynamic_layer_%d' % curr_layer_index)
            new_features = layer(curr_output)
            arr_new_features.append(new_features)
            backward_connections = self.dict_backward_connections_by_layer['dynamic_layer_%d' % curr_layer_index]
            curr_output = [arr_new_features[backward_index] for backward_index in backward_connections]
            curr_output.append(new_features)
            curr_output = torch.cat(curr_output, 1)

        return curr_output

    def __repr__(self):
        str_repr = super(_DynamicBlock, self).__repr__() + os.linesep
        str_connection = self.dict_connections_by_layer['dynamic_layer_0']
        str_repr += 'input connection: {}'.format(str_connection)
        str_repr += os.linesep
        for i in range(self.num_layers):
            curr_layer_index = i + 1
            str_connection = self.dict_connections_by_layer['dynamic_layer_%d' % curr_layer_index]
            str_repr += 'dynamic layer - {}'.format(curr_layer_index)
            str_repr += os.linesep
            str_repr += 'layer connection: {}'.format(str_connection)
            str_repr += os.linesep

        return str_repr

    def _parse_binary_connections(self):
        if isinstance(self._connections[0], list):
            cat_connections = []
            for i in range(len(self._connections)):
                cat_connections += self._connections[i]
            self._connections = cat_connections
        # split the binary connection string into a dictionary with the the layer as the key
        # use output index, starting from 0
        # store the output index which the current output is connected to
        start_index = 0
        for i in range(self.num_layers):
            end_index = start_index + self.num_layers - i
            self._dict_connections_by_layer['dynamic_layer_%d' % i] = self._connections[start_index:end_index]
            start_index = end_index
        self._dict_connections_by_layer['dynamic_layer_%d' % self.num_layers] = []

        # extract backward connections from forward connections
        # use output index, starting from 1
        # store the output index that the current output is connected from
        self._dict_backward_connections_by_layer['dynamic_layer_0'] = []
        for i in range(self.num_layers):
            backward_connections = []
            curr_layer_index = i + 1
            for j in range(curr_layer_index):
                connections = self._dict_connections_by_layer['dynamic_layer_%d' % j]
                if int(connections[curr_layer_index - j - 1]) == 1:
                    backward_connections.append(j)
            self._dict_backward_connections_by_layer['dynamic_layer_%d' % curr_layer_index] = backward_connections

    @property
    def connections(self):
        return self._connections

    @property
    def dict_connections_by_layer(self):
        return self._dict_connections_by_layer

    @property
    def dict_backward_connections_by_layer(self):
        return self._dict_backward_connections_by_layer

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
