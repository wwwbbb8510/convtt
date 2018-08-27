import torch
import numpy as np
from numpy import ndarray
import logging
from torch import nn
from torch.nn import Module
from torch.utils.data import DataLoader
from torch.autograd import Variable
from datetime import datetime
import os


class BaseDriver(object):
    """
    The base class of drivers used to train networks
    """

    def __init__(self, training_epoch=None, batch_size=None, training_data=None, training_label=None,
                 validation_data=None, validation_label=None, test_data=None, test_label=None,
                 gpu_ids=None, optimiser=None):
        """
        The Base driver of training network
        :param training_epoch: the training epoch number
        :type training_epoch: int
        :param batch_size: batch size for training
        :type batch_size: int
        :param training_data: data in the training dataset
        :type training_data: ndarray
        :param training_label: labels in the training dataset
        :type training_label: ndarray
        :param validation_data: data in the validation dataset
        :type validation_data: ndarray
        :param validation_label: labels in the validation dataset
        :type validation_label: ndarray
        :param test_data: data in the test dataset
        :type test_data: ndarray
        :param test_label: labels in the test dataset
        :type test_label: ndarray
        :param gpu_ids: a list of gpu IDs that are going to be used to train the model
        :type gpu_ids: list
        :param optimiser: the optimiser of training the network
        """
        self._training_epoch = training_epoch
        self._batch_size = batch_size
        self._training_data = training_data
        self._training_label = training_label
        self._validation_data = validation_data
        self._validation_label = validation_label
        self._test_data = test_data
        self._test_label = test_label
        self._gpu_ids = gpu_ids
        self._optimiser = optimiser

    def train_model(self):
        raise NotImplementedError()

    def test_model(self, data_loader):
        raise NotImplementedError()

    @property
    def training_epoch(self):
        return self._training_epoch

    @training_epoch.setter
    def training_epoch(self, training_epoch):
        self._training_epoch = training_epoch

    @property
    def batch_size(self):
        return self._batch_size

    @batch_size.setter
    def batch_size(self, batch_size):
        self._batch_size = batch_size

    @property
    def training_data(self):
        return self._training_data

    @training_data.setter
    def training_data(self, training_data):
        self._training_data = training_data

    @property
    def training_label(self):
        return self._training_label

    @training_label.setter
    def training_label(self, training_label):
        self._training_label = training_label

    @property
    def validation_data(self):
        return self._validation_data

    @validation_data.setter
    def validation_data(self, validation_data):
        self._validation_data = validation_data

    @property
    def validation_label(self):
        return self._validation_label

    @validation_label.setter
    def validation_label(self, validation_label):
        self._validation_label = validation_label

    @property
    def test_data(self):
        return self._test_data

    @test_data.setter
    def test_data(self, test_data):
        self._test_data = test_data

    @property
    def test_label(self):
        return self._test_label

    @test_label.setter
    def test_label(self, test_label):
        self._test_label = test_label

    @property
    def gpu_ids(self):
        return self._gpu_ids

    @gpu_ids.setter
    def gpu_ids(self, gpu_ids):
        self._gpu_ids = gpu_ids

    @property
    def optimiser(self):
        return self._optimiser

    @optimiser.setter
    def optimiser(self, optimiser):
        self._optimiser = optimiser

    def __repr__(self):
        str_repr = super(BaseDriver, self).__repr__() + os.linesep
        str_repr += 'Batch size:{}'.format(self.batch_size)
        str_repr += os.linesep
        str_repr += 'Optimiser:{}'.format(self.optimiser)
        str_repr += os.linesep
        str_repr += 'GPUs:{}'.format(self.gpu_ids)
        str_repr += os.linesep
        str_repr += 'Training epoch:{}'.format(self.training_epoch)
        str_repr += os.linesep
        return str_repr


class TorchDriver(BaseDriver):
    """
    The pytorch driver used to train networks
    """

    def __init__(self, model=None, **kwargs):
        """
        initialise
        :param kwargs: key-value parameters which are passed to the parent initialisation method
        """
        super(TorchDriver, self).__init__(**kwargs)

        # torch driver properties
        self._model = model
        self._training_loader = None
        self._validation_loader = None
        self._test_loader = None
        self._test_batch_size = None

        # enable specified gpus
        if self.gpu_ids is not None:
            os.environ['CUDA_VISIBLE_DEVICES'] = '{}'.format(self.gpu_ids)
        # check whether to use cuda
        self._use_cuda = torch.cuda.is_available() and self.gpu_ids is not None

    def train_model(self, test_per_epoch=False):
        """
        train the model
        :return: the trained model
        :rtype: Module
        """
        logging.info('===start training -%s===', self.model)
        if self._use_cuda:
            self.model.cuda()

        # Loss function
        criterion = nn.CrossEntropyLoss()

        # train the model
        epoch_steps = len(self.training_loader)
        for epoch in range(self.training_epoch):  # loop over the dataset multiple times
            running_loss = 0.0
            for i, data in enumerate(self.training_loader, 0):
                # get the inputs
                inputs, labels = data
                inputs, labels = (Variable(inputs), Variable(labels.type(torch.LongTensor)))
                if self._use_cuda:
                    inputs, labels = inputs.cuda(), labels.cuda()
                # zero the parameter gradients
                self.optimiser.zero_grad()
                # forward + backward + optimize
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                self.optimiser.step()
                # print statistics
                running_loss += loss.data[0]
            logging.debug('epoch:{}, training_loss:{}'.format(epoch + 1, running_loss / epoch_steps))
            # Test the model every epoch on validation set
            if self.validation_loader is not None:
                mean_validation_accu, stddev_validation_acccu = self.test_model(self.validation_loader)
                logging.debug(
                    '{}, validation_acc_mean:{}, validation_acc_stddev'.format(datetime.now(), mean_validation_accu,
                                                                               stddev_validation_acccu))
            # Test the model every epoch on test set if needed
            if test_per_epoch:
                mean_test_accu, stddev_test_acccu = self.test_model(self.test_loader)
                logging.debug(
                    '{}, test_acc_mean:{}, test_acc_stddev'.format(datetime.now(), mean_test_accu,
                                                                               stddev_test_acccu))

        return self.model

    def test_model(self, data_loader=None):
        """
        test the model
        :param data_loader: data loader that is used to test the model
        :type data_loader: DataLoader
        :return: (the accuracy mean, the accuracy standard deviation)
        :rtype: tuple
        """
        if data_loader is None:
            data_loader = self.test_loader
        # set model to eval mode
        self.model.eval()
        acc_list = []
        for data in data_loader:
            images, labels = data
            images = Variable(images, volatile=True)
            if self._use_cuda:
                images = images.cuda()
                labels = labels.cuda()
            outputs = self.model(images)
            _, predicted = torch.max(outputs.data, 1)
            total = labels.size(0)
            correct = (predicted.type(torch.LongTensor) == labels.type(torch.LongTensor)).sum()
            acc_list.append(correct / total)
        # set model back to training mode
        self.model.train()
        mean_accu = np.mean(acc_list)
        stddev_acccu = np.std(acc_list)
        logging.debug(
            '{}, test_acc_mean:{}, test_acc_stddev'.format(datetime.now(), mean_accu,
                                                                       stddev_acccu))
        return mean_accu, stddev_acccu

    @property
    def training_loader(self):
        """
        property training_loader
        :return: torch training loader
        :rtype: DataLoader
        """
        if self._training_loader is None:
            self._training_loader = None if self.training_data is None else \
                DataLoader(
                    dataset=torch.utils.data.TensorDataset(torch.from_numpy(self.training_data.astype(np.float32)),
                                                           torch.from_numpy(self.training_label.astype(np.float32))),
                    batch_size=self.batch_size, shuffle=True)
        return self._training_loader

    @property
    def validation_loader(self):
        """
        property validation_loader
        :return: torch validation loader
        :rtype: DataLoader
        """
        if self._validation_loader is None:
            self._validation_loader = None if self.validation_data is None else \
                DataLoader(
                    dataset=torch.utils.data.TensorDataset(torch.from_numpy(self.validation_data.astype(np.float32)),
                                                           torch.from_numpy(self.validation_label.astype(np.float32))),
                    batch_size=self.test_batch_size, shuffle=True)
        return self._validation_loader

    @property
    def test_loader(self):
        """
        property test_loader
        :return: torch test loader
        :rtype: DataLoader
        """
        if self._test_loader is None:
            self._test_loader = None if self.test_data is None else \
                DataLoader(
                    dataset=torch.utils.data.TensorDataset(torch.from_numpy(self.test_data.astype(np.float32)),
                                                           torch.from_numpy(self.test_label.astype(np.float32))),
                    batch_size=self.test_batch_size, shuffle=True)
        return self._test_loader

    @property
    def test_batch_size(self):
        if self._test_batch_size is None:
            self._test_batch_size = self._test_batch_size if self._test_batch_size is not None else self.batch_size
        return self._test_batch_size

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, model):
        self._model = model
