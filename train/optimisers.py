from torch import optim
from torch.nn import Module
from torch.optim.lr_scheduler import MultiStepLR, _LRScheduler
import os
from bisect import bisect_right


class BaseOptimiserWrapper(object):
    """
    Optimiser wrapper base class
    """

    def __init__(self, model, **kwargs):
        """
        Initialise
        :param model: the network model
        :type model: Module
        :param kwargs: the key-value parameters that are passed to the specific optimisers
        """
        self._model = model
        self._optimiser = None
        self._scheduler = None
        # store kwargs in order to reuse
        self._kwargs = kwargs
        self._init_optimiser(**kwargs)

    def step(self):
        """
        move to the next step during the training process
        :return: whether it is applicable to the specific optimiser
        :rtype: bool
        """
        if self._optimiser is not None:
            self._optimiser.step()
            return True
        else:
            return False

    def epoch(self):
        """
        move to the next epoch during the training process
        :return: whether it is applicable to the specific optimiser
        :rtype: bool
        """
        if self._scheduler is not None:
            self._scheduler.step()
            return True
        else:
            return False

    def _init_optimiser(self, **kwargs):
        """
        abstract method: initialise the specific optimiser
        :param kwargs: parameters that are passed to the specific optimiser
        """
        raise NotImplementedError()

    def __getattr__(self, item):
        return getattr(self._optimiser, item)

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, model):
        self._model = model
        self._init_optimiser(**self._kwargs)

    def __repr__(self):
        str_repr = super(BaseOptimiserWrapper, self).__repr__() + os.linesep
        if self._optimiser is not None:
            str_repr += 'Optimiser object:'.format(repr(self._optimiser))
            str_repr += os.linesep
        if self._scheduler is not None:
            str_repr += 'Scheduler object:'.format(repr(self._scheduler))
            str_repr += os.linesep

        return str_repr


class AdamOptimiserWrapper(BaseOptimiserWrapper):
    """
    The wrapper of Adam Optimiser
    """

    def __init__(self, **kwargs):
        super(AdamOptimiserWrapper, self).__init__(**kwargs)

    def _init_optimiser(self, **kwargs):
        self._optimiser = optim.Adam(self.model.parameters(), **kwargs)

    def __repr__(self):
        str_repr = super(AdamOptimiserWrapper, self).__repr__() + os.linesep
        str_repr += 'Optimiser used: Adam Optimiser'
        str_repr += os.linesep

        return str_repr


class ScheduledSGDOptimiserWrapper(BaseOptimiserWrapper):
    """
    The wrapper of Scheduled Stochastic Gradient Decent Optimiser
    """

    def __init__(self, milestones, **kwargs):
        self._milestones = milestones
        super(ScheduledSGDOptimiserWrapper, self).__init__(**kwargs)

    def _init_optimiser(self, **kwargs):
        self._optimiser = optim.SGD(self.model.parameters(), **kwargs)
        self._scheduler = MultiStepLR(self._optimiser, milestones=self.milestones)

    @property
    def milestones(self):
        return self._milestones

    def __repr__(self):
        str_repr = super(ScheduledSGDOptimiserWrapper, self).__repr__() + os.linesep
        str_repr += 'Optimiser used: Scheduled SGD Optimiser'
        str_repr += os.linesep

        return str_repr


class CustomLrsSGDOptimiserWrapper(BaseOptimiserWrapper):
    """
    The wrapper of Scheduled Stochastic Gradient Decent Optimiser
    """

    def __init__(self, lrs, total_epochs=300, **kwargs):
        self._lrs = lrs
        self._total_epochs = total_epochs
        super(CustomLrsSGDOptimiserWrapper, self).__init__(**kwargs)

    def _init_optimiser(self, **kwargs):
        self._optimiser = optim.SGD(self.model.parameters(), **kwargs)
        self._scheduler = MultiStepCustomLRS(self._optimiser, lrs=self._lrs, total_epochs=self._total_epochs)

    @property
    def lrs(self):
        return self._lrs

    @property
    def total_epochs(self):
        return self._total_epochs

    def __repr__(self):
        str_repr = super(CustomLrsSGDOptimiserWrapper, self).__repr__() + os.linesep
        str_repr += 'Optimiser used: Custom LRS(learning rates) SGD Optimiser'
        str_repr += os.linesep

        return str_repr


class MultiStepCustomLRS(_LRScheduler):
    """Set the learning rate of each parameter group to customized learning rates
    last_epoch=-1, sets initial lr as lr.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        lrs (list): List of customized learning rates.
        total_epochs (float): total number of training epochs.
        last_epoch (int): The index of last epoch. Default: -1.

    Example:
        >>> scheduler = MultiStepCustomLRS(optimizer, lrs=[0.1, 0.01, 0.001], total_epochs=300)
        >>> for epoch in range(100):
        >>>     scheduler.step()
        >>>     train(...)
        >>>     validate(...)
    """

    def __init__(self, optimizer, lrs, total_epochs=300, last_epoch=-1):
        self.lrs = lrs
        self.total_epochs = total_epochs
        num_milestones = len(lrs) - 1
        self._milestones = [i * (self.total_epochs / (num_milestones + 1)) for i in range(1, num_milestones+1)]
        super(MultiStepCustomLRS, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        return [self.lrs[bisect_right(self._milestones, self.last_epoch)]
                for base_lr in self.base_lrs]
