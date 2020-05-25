import numpy as np


class LossFunction:

    def calculate(self, prediction, target, one_hot_encoding):
        raise NotImplementedError

    def derivative(self, prediction, target):
        raise NotImplementedError


class AbsoluteError(LossFunction):

    def __str__(self):
        return "AbsoluteError"

    def calculate(self, prediction, target, one_hot_encoding):
        loss = np.sum(np.abs(prediction - target))
        return loss / len(target)

    def derivative(self, prediction, target):
        return np.sign(prediction - target)


class SquaredError(LossFunction):

    def __str__(self):
        return "SquaredError"

    def calculate(self, prediction, target, one_hot_encoding):
        loss = np.sum((prediction - target) ** 2)
        return loss / (2 * len(target))

    def derivative(self, prediction, target):
        return prediction - target


# class Hinge(LossFunction):
#     """Binary"""
#
#     def __str__(self):
#         return "Hinge (binary)"
#
#     def calculate(self, prediction, target, one_hot_encoding):
#         dist = 1 - prediction * target
#         loss = np.sum(dist * (dist > 0))
#         return loss / len(target)
#
#     def derivative(self, prediction, target):
#         return -prediction * ((1 - prediction * target) < 1)
#
#
# class SquaredHinge(LossFunction):
#     """Binary"""
#
#     def __str__(self):
#         return "SquaredHinge (binary)"
#
#     def calculate(self, prediction, target, one_hot_encoding):
#         dist = 1 - prediction * target
#         loss = np.sum((dist * (dist > 0)) ** 2)
#         return loss / (2 * len(target))
#
#     def derivative(self, prediction, target):
#         p_t = prediction * target
#         return (- p_t * ((-p_t) > 0)) * (-prediction * ((1 - p_t) < 1))


class CrossEntropy(LossFunction):
    """Categorial"""

    def __str__(self):
        return "CrossEntropy"

    def calculate(self, prediction, target, one_hot_encoding):
        loss = -np.sum(target * np.log(prediction + 1e-9) + (1 - target) * np.log(1 - prediction + 1e-9))
        return loss / len(target)

    def derivative(self, prediction, target):
        return -(target / (prediction + 1e-9) - (1 - target) / (1 - prediction + 1e-9))

# class LogCosh(LossFunction):
#
#     def __str__(self):
#         return 'LogCosh'
#
#     def calculate(self, prediction, target, one_hot_encoding):
#         pass
#
#     def derivative(self, prediction, target):
#         pass
