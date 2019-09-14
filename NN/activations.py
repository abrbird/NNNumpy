import numpy as np


class ActivationFunction:

    def __str__(self):
        raise NotImplementedError

    def calculate(self, x):
        raise NotImplementedError

    def derivative(self, x):
        raise NotImplementedError


class Linear(ActivationFunction):

    def __str__(self):
        return 'Linear(-inf,inf)'

    def calculate(self, x):
        return x

    def derivative(self, x):
        return np.ones_like(x)


class TanH(ActivationFunction):

    def __str__(self):
        return 'TanH(-1,1)'

    def calculate(self, x):
        e_p_2x = np.exp(2 * x)
        return (e_p_2x - 1) / (e_p_2x + 1)

    def derivative(self, x):
        return 1 - self.calculate(x) ** 2


class ReLU(ActivationFunction):

    def __str__(self):
        return 'ReLU[0,inf)'

    def calculate(self, x):
        return x * (x > 0)

    def derivative(self, x):
        return 1. * (x > 0)


class SoftPlus(ActivationFunction):

    def __str__(self):
        return 'SoftPlus(0,inf)'

    def calculate(self, x):
        return np.log(1 + np.exp(x))

    def derivative(self, x):
        return 1. / (1. + np.exp(-x))


class Sigmoid(ActivationFunction):

    def __str__(self):
        return 'Sigmoid(0,1)'

    def calculate(self, x):
        return 1. / (1. + np.exp(-x))

    def derivative(self, x):
        sigm_x = self.calculate(x)
        return sigm_x * (1 - sigm_x)


class SoftMax(ActivationFunction):
    """ Only for 2D matrix!!!"""

    def __str__(self):
        return 'SoftMax(probabilities)'

    def calculate(self, x):
        exps = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exps / np.sum(exps, axis=1, keepdims=True)

    def derivative(self, x):
        sms = self.calculate(x)
        return sms * (1 - sms)


class ArcTan(ActivationFunction):

    def __str__(self):
        return 'ArcTan(-Pi/2,Pi/2)'

    def calculate(self, x):
        return np.arctan(x)

    def derivative(self, x):
        return 1 / (x ** 2 + 1)


class ISRU(ActivationFunction):

    def __str__(self):
        return 'ISRU(-1,1)'

    def calculate(self, x):
        return x / np.sqrt(1 + x ** 2)

    def derivative(self, x):
        return (1 / np.sqrt(1 + x ** 2)) ** 3


class BentIdentity(ActivationFunction):

    def __str__(self):
        return 'BentIdentity(-inf,inf)'

    def calculate(self, x):
        return x + 0.5 * (np.sqrt(x ** 2 + 1) - 1)

    def derivative(self, x):
        return 1 + (x / (2 * np.sqrt(x ** 2 + 1)))


class Sinusoid(ActivationFunction):

    def __str__(self):
        return "Sinusoid[-1,1]"

    def calculate(self, x):
        return np.sin(x)

    def derivative(self, x):
        return np.cos(x)


class Gaussian(ActivationFunction):

    def __str__(self):
        return 'Gaussian(0,1]'

    def calculate(self, x):
        return np.exp(-x ** 2)

    def derivative(self, x):
        return -2 * x * self.calculate(x)


activations_names = {"Linear": Linear,
                     "TanH": TanH,
                     "ReLU": ReLU,
                     "SoftPlus": SoftPlus,
                     "Sigmoid": Sigmoid,
                     "SoftMax": SoftMax,
                     "ArcTan": ArcTan,
                     "ISRU": ISRU,
                     "BentIdentity": BentIdentity,
                     "Sinusoid": Sinusoid,
                     "Gaussian": Gaussian
                     }

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    for activation in activations_names.values():
        if activation != SoftMax:
            f = activation()
            linspace = np.linspace(-5, 5, 200)
            activation = f.calculate(linspace)
            plt.plot(linspace, activation)
            plt.title(str(f))
            plt.show()
