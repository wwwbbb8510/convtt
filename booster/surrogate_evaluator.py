import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import logging


class LSTMTrainer(object):
    '''
    lstm trainer - a surrogate model to predict the accuracy
    given the accuracies of the first few steps
    and a training dataset containing the accuracies of many training processes
    '''

    def __init__(self, seed=0, future_steps=100, model=None, training_input=None, training_target=None,
                 validation_input=None, validation_target=None):
        '''
        init
        :param seed: random seed
        :param future_steps: the number of future steps where the prediction needs to perform
        :param model: the lstm model
        '''
        self.model = LSTMTrainerModel() if model is None else model
        self.seed = seed

        # training, validation, test data
        self._training_input = training_input
        self._training_target = training_target
        self._validation_input = validation_input
        self._validation_target = validation_target

        # set the predicted steps
        self._future_steps = future_steps

    def train(self):
        '''
        train the lstm model
        :return:
        '''
        # set random seed
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        training_input = torch.from_numpy(self.training_input)
        training_target = torch.from_numpy(self.training_target)
        validation_input = torch.from_numpy(self.validation_input)
        validation_target = torch.from_numpy(self.validation_target)
        # build model and loss
        self.model.double()
        criterion = nn.MSELoss()
        # use LBFGS as optimizer since we can load the whole data to train
        optimizer = optim.LBFGS(self.model.parameters(), lr=0.8)
        # begin to train
        for i in range(15):
            logging.debug('STEP: {}'.format(i))

            def closure():
                optimizer.zero_grad()
                out = self.model(training_input)
                loss = criterion(out, training_target)
                logging.debug('loss: {}'.format(loss.item()))
                loss.backward()
                return loss

            optimizer.step(closure)
            # begin to predict, no need to track gradient here
            with torch.no_grad():
                pred = self.model(validation_input, future=self.future_steps)
                loss = criterion(pred[:, :-self.future_steps], validation_target)
                logging.debug('test loss: {}'.format(loss.item()))
                y = pred.detach().numpy()

    def predict(self, test_input):
        '''
        predict the target given the input
        :param test_input: the input accuracies of the first few steps
        :return: the predicted target accuracies
        '''
        test_input = torch.from_numpy(test_input)
        with torch.no_grad():
            pred = self.model(test_input, future=self.future_steps)
            predicted_target = pred.detach().numpy()
        return predicted_target

    @property
    def training_input(self):
        return self._training_input

    @training_input.setter
    def training_input(self, training_input):
        self._training_input = training_input

    @property
    def training_target(self):
        return self._training_target

    @training_target.setter
    def training_target(self, training_target):
        self._training_target = training_target

    @property
    def validation_input(self):
        return self._validation_input

    @validation_input.setter
    def validation_input(self, validation_input):
        self._validation_input = validation_input

    @property
    def validation_target(self):
        return self._validation_target

    @validation_target.setter
    def validation_target(self, validation_target):
        self._validation_target = validation_target

    @property
    def test_input(self):
        return self._test_input

    @test_input.setter
    def test_input(self, test_input):
        self._test_input = test_input

    @property
    def test_target(self):
        return self._test_target

    @test_target.setter
    def test_target(self, test_target):
        self._test_target = test_target

    @property
    def future_steps(self):
        return self._future_steps

    @future_steps.setter
    def future_steps(self, future_steps):
        self._future_steps = future_steps

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, model):
        self._model = model


class LSTMTrainerModel(nn.Module):
    def __init__(self):
        super(LSTMTrainerModel, self).__init__()
        self.lstm1 = nn.LSTMCell(1, 51)
        self.lstm2 = nn.LSTMCell(51, 51)
        self.linear = nn.Linear(51, 1)

    def forward(self, input, future=0):
        outputs = []
        h_t = torch.zeros(input.size(0), 51, dtype=torch.double)
        c_t = torch.zeros(input.size(0), 51, dtype=torch.double)
        h_t2 = torch.zeros(input.size(0), 51, dtype=torch.double)
        c_t2 = torch.zeros(input.size(0), 51, dtype=torch.double)

        for i, input_t in enumerate(input.chunk(input.size(1), dim=1)):
            h_t, c_t = self.lstm1(input_t, (h_t, c_t))
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
            output = self.linear(h_t2)
            outputs += [output]
        for i in range(future):  # if we should predict the future
            h_t, c_t = self.lstm1(output, (h_t, c_t))
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
            output = self.linear(h_t2)
            outputs += [output]
        outputs = torch.stack(outputs, 1).squeeze(2)
        return outputs


if __name__ == '__main__':
    # load data
    data = torch.load('log/traindata.pt')

    lstm_trainer = LSTMTrainer()
    lstm_trainer.training_input, lstm_trainer.training_target, lstm_trainer.validation_input, lstm_trainer.validation_target = \
        data[3:, :-1], data[3:, 1:], data[:3, :-1], data[:3, 1:]
    lstm_trainer.train()
    test_input = None
    lstm_trainer.predict(test_input)
