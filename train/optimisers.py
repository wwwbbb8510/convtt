from torch import optim
from torch.nn import Module
from torch.optim.lr_scheduler import MultiStepLR
import os


class BaseOptimiserWrapper(object):
    """
    Optimiser wrapper base class
    """

    def __init__(self, model, **kwargs):
        """
        Initialise
        :param model: the network model
        :type model: Module
        :param kwargs: the kye-value parameters that are passed to the specific optimisers
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
