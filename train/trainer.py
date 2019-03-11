from convtt.train.drivers import TorchDriver
from convtt.train.optimisers import BaseOptimiserWrapper
from torch.nn import Module
import os

__all__ = ['build_driver', 'build_optimiser', 'build_trainer']


class BaseTrainer(object):
    def __init__(self, driver=None, optimiser=None):
        """
        initialise
        :param driver: Training driver
        :type driver: TorchDriver
        :param optimiser: Training optimiser
        :type optimiser: BaseOptimiserWrapper
        """
        super(BaseTrainer, self).__init__()
        self._driver = driver
        self._optimiser = optimiser

    def eval(self, model=None):
        raise NotImplementedError()

    @property
    def driver(self):
        return self._driver

    @driver.setter
    def driver(self, driver):
        self._driver = driver

    @property
    def optimiser(self):
        return self._optimiser

    @optimiser.setter
    def optimiser(self, optimiser):
        self._optimiser = optimiser

    def __repr__(self):
        str_repr = super(BaseTrainer, self).__repr__() + os.linesep
        str_repr += 'driver:{}'.format(self.driver)
        str_repr += os.linesep
        str_repr += 'optimiser:{}'.format(self.optimiser)
        str_repr += os.linesep
        return str_repr


class TorchTrainer(BaseTrainer):
    def __init__(self, **kwargs):
        super(TorchTrainer, self).__init__(**kwargs)

    def eval(self, model=None):
        if model is not None:
            self.driver.model = model
        if model is not None:
            self.optimiser.model = model
        self.driver.train_model()
        return self.driver.test_model()


def build_optimiser(name, **kwargs):
    """
    build an optimiser
    :param name: the name of optimiser
    :type name: str
    :param kwargs: the arguments that are passed to optimiser
    :return: the created optimiser wrapper
    :rtype: BaseOptimiserWrapper
    """
    optimiser = None
    if name.lower() == 'adam':
        from convtt.train.optimisers import AdamOptimiserWrapper
        optimiser = AdamOptimiserWrapper(**kwargs)
    elif name.lower() == 'scheduledsgd':
        from convtt.train.optimisers import ScheduledSGDOptimiserWrapper
        optimiser = ScheduledSGDOptimiserWrapper(**kwargs)
    elif name.lower() == 'customlrssgd':
        from convtt.train.optimisers import CustomLrsSGDOptimiserWrapper
        optimiser = CustomLrsSGDOptimiserWrapper(**kwargs)

    return optimiser


def build_driver(**kwargs):
    """
    build a driver
    :param kwargs: parameters that are passed to driver
    :return: a training driver
    :rtype: TorchDriver
    """
    driver = TorchDriver(**kwargs)
    return driver


def build_trainer(driver, optimiser):
    """
    build a trainer
    :param driver: driver
    :type driver: TorchDriver
    :param optimiser: optimiser
    :type optimiser: BaseOptimiserWrapper
    :return: a trainer
    :rtype: TorchTrainer
    """
    trainer = TorchTrainer(driver=driver, optimiser=optimiser)
    return trainer
