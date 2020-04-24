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
                 gpu_ids=None, optimiser=None, early_stop_max_epochs=10):
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
        :param early_stop_max_epochs: the training stops when it won't improve in early_stop_max_epochs epochs
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
        self.early_stop_max_epochs = early_stop_max_epochs

    def train_model(self):
        raise NotImplementedError()

    def test_model(self, data_loader):
        raise NotImplementedError()

    def save_model(self, file_path):
        raise NotImplementedError()

    def load_model(self, file_path):
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
        self._best_validation_acc = None
        self._training_loss_history = None
        self._validation_acc_history = None
        self._best_validation_epoch = None

        # enable specified gpus
        if self.gpu_ids is not None:
            os.environ['CUDA_VISIBLE_DEVICES'] = '{}'.format(self.gpu_ids)
        # check whether to use cuda
        self._use_cuda = torch.cuda.is_available()

    def train_model(self, test_per_epoch=False, topk=(1,), eval_training_set=True, gpu_id=None, use_sampler=False):
        """
        train the model
        :return: the trained model
        :rtype: Module
        """
        logging.info('===start training -%s===', self.model)
        if self._use_cuda:
            self.model.cuda() if gpu_id is None else self.model.cuda(gpu_id)

        # Loss function
        criterion = nn.CrossEntropyLoss()
        if self._use_cuda:
            criterion.cuda() if gpu_id is None else criterion.cuda(gpu_id)

        # train the model
        epoch_steps = len(self.training_loader)
        self._best_validation_acc = 0
        self._best_validation_epoch = 0
        self._training_loss_history = []
        self._validation_acc_history = []
        for epoch in range(self.training_epoch):  # loop over the dataset multiple times
            self.optimiser.epoch()
            logging.debug(
                'training epoch {} --- lr : {}'.format(epoch, self.optimiser._optimiser.param_groups[0]['lr']))
            running_loss = 0.0
            if use_sampler:
                self.training_loader.sampler.set_epoch(epoch)
                logging.debug('training epoch {} --- set epoch as the random seed for training sampler'.format(epoch))
            for i, data in enumerate(self.training_loader, 0):
                # get the inputs
                inputs, labels = data
                inputs, labels = Variable(inputs), Variable(labels.type(torch.LongTensor))
                if self._use_cuda:
                    inputs, labels = (inputs.cuda(), labels.cuda()) if gpu_id is None else \
                        (inputs.cuda(gpu_id, non_blocking=True), labels.cuda(gpu_id, non_blocking=True))
                # zero the parameter gradients
                self.optimiser.zero_grad()
                # forward + backward + optimize
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                self.optimiser.step()
                # print statistics
                running_loss += loss.item()
                logging.debug('training progress --- epoch: {}, step: {}'.format(epoch, i)) if i % 100 == 0 else None
            logging.debug('epoch:{}, training_loss:{}'.format(epoch + 1, running_loss / epoch_steps))
            self._training_loss_history.append(running_loss / epoch_steps)
            # Test the model every epoch on validation set along with outputting training accuracy
            if self.validation_loader is not None:
                if eval_training_set:
                    mean_training_accu, stddev_training_acccu = self.test_model(self.training_loader, topk, gpu_id)
                    self.print_topk_acc('training_acc', mean_training_accu, stddev_training_acccu, topk)
                mean_validation_accu, stddev_validation_acccu = self.test_model(self.validation_loader, topk, gpu_id)
                self.print_topk_acc('validation_acc', mean_validation_accu, stddev_validation_acccu, topk)
                self._validation_acc_history.append(mean_validation_accu[0])

            # Test the model every epoch on test set if needed
            if test_per_epoch:
                mean_test_accu, stddev_test_acccu = self.test_model(self.test_loader, topk, gpu_id)
                self.print_topk_acc('test_acc', mean_test_accu, stddev_test_acccu, topk)

            if self._best_validation_acc < mean_validation_accu[0]:
                self._best_validation_acc = mean_validation_accu[0]
                self._best_validation_epoch = epoch
            else:
                if epoch - self._best_validation_epoch >= self.early_stop_max_epochs:
                    break

        return self.model

    def print_topk_acc(self, label_prefix, mean_acc, stddev_acc, topk=(1,)):
        """
        print topk accuracy
        :param label_prefix: label prefix: training, validation, test
        :type label_prefix: str
        :param mean_acc: topk mean accuracy
        :type mean_acc: list
        :param stddev_acc: topk accuracy standard deviation
        :type stddev_acc: list
        :param topk: topk config to define top-n accuracies that need to be calculated
        :type topk: tuple
        """
        for i in range(len(topk)):
            if topk[i] == 1:
                logging.debug(
                    '{}, {}_mean:{}, {}_stddev:{}'.format(datetime.now(), label_prefix, mean_acc[i],
                                                          label_prefix,
                                                          stddev_acc[i]))
            else:
                logging.debug(
                    '{}, {}_mean_top{}:{}, {}_stddev_top{}:{}'.format(datetime.now(), label_prefix, topk[i],
                                                                      mean_acc[i],
                                                                      label_prefix, topk[i],
                                                                      stddev_acc[i]))

    def test_model(self, data_loader=None, topk=(1,), gpu_id=None):
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
        with torch.no_grad():
            for data in data_loader:
                images, labels = data
                if self._use_cuda:
                    images = images.cuda() if gpu_id is None else images.cuda(gpu_id, non_blocking=True)
                    labels = labels.cuda() if gpu_id is None else labels.cuda(gpu_id, non_blocking=True)
                outputs = self.model(images)
                acc_topk = self.calculate_accuracy(outputs, labels, topk)
                acc_list.append(acc_topk)
        # set model back to training mode
        self.model.train()
        mean_accu = np.mean(acc_list, axis=0)
        stddev_acccu = np.std(acc_list, axis=0)
        return mean_accu, stddev_acccu

    def calculate_accuracy(self, outputs, labels, topk=(1,)):
        """Computes the accuracy over the k top predictions for the specified values of k"""
        with torch.no_grad():
            maxk = max(topk)
            batch_size = outputs.size(0)

            _, pred = outputs.topk(maxk, 1, True, True)
            pred = pred.t()
            correct = pred.eq(labels.view(1, -1).expand_as(pred))

            res = []
            for k in topk:
                correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
                res.append((correct_k.mul_(1 / batch_size)[0]).item())
            return res

    def save_model(self, file_path):
        torch.save(self.model.state_dict(), file_path)

    def load_model(self, file_path):
        self.model.load_state_dict(torch.load(file_path))

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
        """
        model getter
        :return: model
        :rtype: Module
        """
        return self._model

    @model.setter
    def model(self, model):
        self._model = model

    @property
    def best_validation_acc(self):
        """
        best_validation_acc getter
        :return: best_validation_acc
        :rtype: Module
        """
        return self._best_validation_acc

    @property
    def best_validation_epoch(self):
        """
        best_validation_epoch getter
        :return: best_validation_epoch
        :rtype: Module
        """
        return self._best_validation_epoch

    @property
    def training_loss_history(self):
        """
        training loss history getter
        which contains all the training losses during the training process
        :return: _training_loss_history
        :rtype: list
        """
        return self._training_loss_history

    @property
    def validation_acc_history(self):
        """
        validation acc history getter
        which contains all the validation accuracies during the training process
        :return: _validation_acc_history
        :rtype: list
        """
        return self._validation_acc_history
