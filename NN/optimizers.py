import numpy as np
from NN.losses import SquaredError
from NN.helper import LRScheduler


class Optimizer:

    def optimize(self, epoch, grads_, grads):
        raise NotImplementedError


class SGD(Optimizer):

    def __str__(self):
        return "SGD"

    def __init__(self, layers, lr_scheduler: LRScheduler = None, loss_function=None):
        self.layers = layers
        self.lr_scheduler = lr_scheduler
        self.loss_function = loss_function

        if self.loss_function is None:
            self.loss_function = SquaredError()
        if self.lr_scheduler is None:
            self.lr_scheduler = LRScheduler(1e-3)

    def optimize(self, epoch, grads_, grads):
        lr = self.lr_scheduler.get_learning_rate(epoch)
        for i in range(len(self.layers) - 1, 0 - 1, -1):
            self.layers[i].update(lr, grads_[i], grads[i])


class NesterovAG(Optimizer):

    def __str__(self):
        return "NesterovAG"

    def __init__(self, layers, lr_scheduler: LRScheduler = None, loss_function=None, opt_multiplier=0.9):
        self.layers = layers
        self.lr_scheduler = lr_scheduler
        self.loss_function = loss_function

        self.betta = opt_multiplier
        self.velocities = [np.zeros_like(layer.get_regularization_params()) for layer in layers]

        if self.loss_function is None:
            self.loss_function = SquaredError()
        if self.lr_scheduler is None:
            self.lr_scheduler = LRScheduler(1e-3)

    def optimize(self, epoch, grads_, grads):
        lr = self.lr_scheduler.get_learning_rate(epoch)
        for i in range(len(self.layers) - 1, - 1, -1):
            self.velocities[i] = self.betta * self.velocities[i] + (1 - self.betta) * grads[i]
            self.layers[i].update(lr, grads_[i], self.velocities[i])


class Adagrad(Optimizer):

    def __str__(self):
        return "Adagrad"

    def __init__(self, layers, lr_scheduler: LRScheduler = None, loss_function=None):
        self.layers = layers
        self.lr_scheduler = lr_scheduler
        self.loss_function = loss_function

        self.accumulateds = [np.zeros_like(layer.get_regularization_params()) for layer in layers]

        if self.loss_function is None:
            self.loss_function = SquaredError()
        if self.lr_scheduler is None:
            self.lr_scheduler = LRScheduler(1e-3)

    def optimize(self, epoch, grads_, grads):
        lr = self.lr_scheduler.get_learning_rate(epoch)
        for i in range(len(self.layers) - 1, - 1, -1):
            self.accumulateds[i] = self.accumulateds[i] + grads[i] ** 2
            grads[i] = grads[i] / (np.sqrt(self.accumulateds[i]) + 1e-12)
            self.layers[i].update(lr, grads_[i], grads[i])


class RMSprop(Optimizer):

    def __str__(self):
        return "RMSprop"

    def __init__(self, layers, lr_scheduler: LRScheduler = None, loss_function=None, opt_multiplier=0.95):
        self.layers = layers
        self.lr_scheduler = lr_scheduler
        self.loss_function = loss_function

        self.gamma = opt_multiplier
        self.accumulateds = [np.zeros_like(layer.get_regularization_params()) for layer in layers]

        if self.loss_function is None:
            self.loss_function = SquaredError()
        if self.lr_scheduler is None:
            self.lr_scheduler = LRScheduler(1e-3)

    def optimize(self, epoch, grads_, grads):
        lr = self.lr_scheduler.get_learning_rate(epoch)
        for i in range(len(self.layers) - 1, - 1, -1):
            self.accumulateds[i] = self.gamma * self.accumulateds[i] + (1 - self.gamma) * grads[i] ** 2
            grads[i] = grads[i] / (np.sqrt(self.accumulateds[i]) + 1e-12)
            self.layers[i].update(lr, grads_[i], grads[i])


class Adam(Optimizer):

    def __str__(self):
        return "Adam"

    def __init__(self, layers, lr_scheduler: LRScheduler = None, loss_function=None, opt_multipliers=(0.9, 0.95)):
        self.layers = layers
        self.lr_scheduler = lr_scheduler
        self.loss_function = loss_function

        self.betta, self.gamma = opt_multipliers
        self.velocities = [np.zeros_like(layer.get_regularization_params()) for layer in layers]
        self.accumulateds = [np.zeros_like(layer.get_regularization_params()) for layer in layers]

        if self.loss_function is None:
            self.loss_function = SquaredError()
        if self.lr_scheduler is None:
            self.lr_scheduler = LRScheduler(1e-3)

    def optimize(self, epoch, grads_, grads):
        lr = self.lr_scheduler.get_learning_rate(epoch)
        for i in range(len(self.layers) - 1, - 1, -1):
            self.velocities[i] = self.betta * self.velocities[i] + (1 - self.betta) * grads[i]
            self.accumulateds[i] = self.gamma * self.accumulateds[i] + (1 - self.gamma) * grads[i] ** 2
            v_ = self.velocities[i] / (1 - self.betta ** epoch)
            a_ = self.accumulateds[i] / (1 - self.gamma ** epoch)
            grads[i] = v_ / (np.sqrt(a_) + 1e-12)
            self.layers[i].update(lr, grads_[i], grads[i])
