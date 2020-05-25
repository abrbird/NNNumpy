import numpy as np


class Regularization:

    def calculate(self, weights):
        raise NotImplementedError

    def derivative(self, weights):
        raise NotImplementedError


class NoRegularization(Regularization):

    def __str__(self):
        return "No regularization"

    def calculate(self, weights):
        return 0

    def derivative(self, weights):
        return 0


class L1Regularization(Regularization):

    def __str__(self):
        return "L1 regularization"

    def __init__(self, l1_multiplier=1e-3):
        self.l1_multiplier = l1_multiplier

    def calculate(self, weights):
        result = self.l1_multiplier * np.sum(np.abs(weights))
        return result

    def derivative(self, weights):
        return self.l1_multiplier * np.sign(weights)


class L2Regularization(Regularization):

    def __str__(self):
        return "L2 regularization"

    def __init__(self, l2_multiplier=1e-3):
        self.l2_multiplier = l2_multiplier

    def calculate(self, weights):
        result = 0.5 * self.l2_multiplier * np.sum(weights ** 2)
        return result

    def derivative(self, weights):
        return self.l2_multiplier * weights


class L1L2Regularization(Regularization):

    def __str__(self):
        return "L1L2 regularization"

    def __init__(self, l1_multiplier=1e-3, l2_multiplier=1e-3):
        self.l1_multiplier = l1_multiplier
        self.l2_multiplier = l2_multiplier

    def calculate(self, weights):
        result = self.l1_multiplier * np.sum(np.abs(weights))
        result += 0.5 * self.l2_multiplier * np.sum(weights**2)
        return result

    def derivative(self, weights):
        return self.l1_multiplier * np.sign(weights) + self.l2_multiplier * weights
