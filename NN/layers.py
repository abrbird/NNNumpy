import numpy as np
from NN.activations import ActivationFunction, Linear, SoftMax
from NN.regularizations import NoRegularization


class Layer:

    def __str__(self):
        raise NotImplementedError

    def __init__(self):
        self.input_shape = None
        self.output_shape = None

    def pre_activate(self, input_signal, is_training=False):
        raise NotImplementedError

    def activate(self, pre_act, is_training=False):
        raise NotImplementedError

    def forward_pass(self, input_signal, is_training=False):
        raise NotImplementedError

    def backward_pass(self, cached_input_signal, cached_pre_act, backward_signal):
        raise NotImplementedError

    def update(self, lr, grad_, grad):
        raise NotImplementedError

    def get_regularization_params(self):
        raise NotImplementedError


class FullyConnected(Layer):

    def __str__(self):
        return "FullyConnected"

    def __init__(self, outputs, activation: ActivationFunction = None, input_shape: tuple = None,
                 regularization=None, weights: np.ndarray = None, bias: np.ndarray = None, connected_to: Layer = None):
        super(FullyConnected).__init__()
        if input_shape is None and connected_to is None:
            raise ValueError('argument input_shape or connected_to must be not None')
        if input_shape:
            if len(input_shape) == 2:
                self.input_shape = input_shape
            else:
                raise ValueError('input_shape must have 2 dimensions.')
        if connected_to:
            if len(connected_to.output_shape) == 2:
                self.input_shape = connected_to.output_shape
            else:
                raise ValueError('output_shape of connected_to layer must have 2 dimensions.')
        self.output_shape = (None, outputs)
        self.activation = activation
        self.regularization = regularization
        self.bias = None
        self.weights = None

        if self.activation is None:
            self.activation = Linear()

        if self.regularization is None:
            self.regularization = NoRegularization()

        if bias is not None and bias.shape[0] == outputs:
            self.bias = bias
        else:
            multiplier = 1 / np.sqrt(self.input_shape[1] / 2)
            self.bias = multiplier * np.random.normal(loc=0, scale=0.1, size=outputs)

        if weights is not None and weights.shape == (self.input_shape[1], outputs):
            self.weights = np.copy(weights)
        else:
            multiplier = 1 / np.sqrt(self.input_shape[1] / 2)
            self.weights = multiplier * np.random.normal(loc=0, scale=1, size=(self.input_shape[1], outputs))

    def pre_activate(self, input_signal, is_training=False):
        return np.dot(input_signal, self.weights) + self.bias

    def activate(self, pre_act, is_training=False):
        return self.activation.calculate(pre_act)

    def forward_pass(self, input_signal, is_training=False):
        return self.activate(self.pre_activate(input_signal, is_training), is_training)

    def backward_pass(self, cached_input_signal, cached_pre_act, backward_signal):
        delta = backward_signal * self.activation.derivative(cached_pre_act)
        next_backward_signal = np.dot(delta, self.weights.T)
        grad_ = np.sum(delta, axis=0)
        grad = np.dot(cached_input_signal.T, delta) + self.regularization.derivative(self.get_regularization_params())
        return next_backward_signal, grad_, grad

    def update(self, lr, step_grad_, step_grad):
        self.weights -= lr * step_grad
        self.bias -= lr * step_grad_

    def get_regularization_params(self):
        return self.weights


class Activation(Layer):

    def __str__(self):
        return "Activation"

    def __init__(self, activation, input_shape: tuple = None, connected_to: Layer = None):
        super(Activation).__init__()
        self.activation = activation
        self.regularization = NoRegularization()
        if input_shape:
            self.input_shape = input_shape
        elif connected_to:
            self.input_shape = connected_to.output_shape
        else:
            raise ValueError('argument input_shape or connected_to must be not None')
        self.output_shape = self.input_shape

    def pre_activate(self, input_signal, is_training=False):
        return input_signal

    def activate(self, pre_act, is_training=False):
        return self.activation.calculate(pre_act)

    def forward_pass(self, input_signal, is_training=False):
        return self.activate(self.pre_activate(input_signal, is_training), is_training)

    def backward_pass(self, cached_input_signal, cached_pre_act, backward_signal):
        delta = backward_signal * self.activation.derivative(cached_pre_act)
        next_backward_signal = delta
        return next_backward_signal, 0, 0

    def update(self, lr, step_grad_, step_grad):
        pass

    def get_regularization_params(self):
        return np.array([0])


class Dropout(Layer):

    def __str__(self):
        return "Dropout"

    def __init__(self, prob=0.5, input_shape: tuple = None, connected_to: Layer = None):
        super(Dropout).__init__()
        self.p = 0.5
        self.regularization = NoRegularization()

        if input_shape:
            self.input_shape = input_shape
        elif connected_to:
            self.input_shape = connected_to.output_shape
        else:
            raise ValueError('argument input_shape or connected_to must be not None')
        self.output_shape = self.input_shape

        if 0 < prob <= 1:
            self.p = prob
        else:
            raise ValueError('An expected value is between 0 and 1. Instead got {}'.format(prob))

    def pre_activate(self, input_signal, is_training=False):
        if is_training:
            mask = (np.random.rand(*input_signal.shape) >= self.p) / self.p
            # mask = np.random.binomial(1, p=self.p, size=input_signal.shape) / self.p
            return input_signal, mask
        else:
            return input_signal

    def activate(self, pre_act, is_training=False):
        if is_training:
            return pre_act[0] * pre_act[1]
        else:
            return pre_act

    def forward_pass(self, input_signal, is_training=False):
        return self.activate(self.pre_activate(input_signal, is_training), is_training)

    def backward_pass(self, cached_input_signal, cached_pre_act, backward_signal):
        delta = backward_signal * cached_pre_act[1]
        next_backward_signal = delta
        grad_ = 0
        grad = 0
        return next_backward_signal, grad_, grad

    def update(self, lr, step_grad_, step_grad):
        pass

    def get_regularization_params(self):
        return 0


class BatchNormalization(Layer):

    def __str__(self):
        return 'BatchNormalization'

    def __init__(self, betta=0, gamma=1, momentum=0.9, epsilon=1e-5, is_trainable=True, regularization=None,
                 input_shape: tuple = None, connected_to: Layer = None):
        super(BatchNormalization).__init__()
        self.mean = 0
        self.var = 1
        self.betta = betta
        self.gamma = gamma
        self.momentum = momentum
        self.epsilon = epsilon
        self.regularization = regularization
        self.is_trainable = is_trainable
        if input_shape:
            self.input_shape = input_shape
        elif connected_to:
            self.input_shape = connected_to.output_shape
        else:
            raise ValueError('argument input_shape or connected_to must be not None')
        self.output_shape = self.input_shape

        if self.regularization is None:
            self.regularization = NoRegularization()

    def pre_activate(self, input_signal, is_training=False):
        if is_training:
            mean_ = np.mean(input_signal, axis=0)
            x_m = input_signal - mean_
            var_ = np.var(input_signal, axis=0)

            self.mean = self.momentum * self.mean + (1 - self.momentum) * mean_
            self.var = self.momentum * self.var + (1 - self.momentum) * var_

            return input_signal, x_m, mean_, var_
        else:
            return input_signal

    def activate(self, pre_act, is_training=False):
        if is_training:
            input_signal, x_m, mean, var = pre_act
            return self.gamma * x_m / np.sqrt(var + self.epsilon) + self.betta
        else:
            return self.gamma * (pre_act - self.mean) / np.sqrt(self.var + self.epsilon) + self.betta

    def forward_pass(self, input_signal, is_training=False):
        return self.activate(self.pre_activate(input_signal, is_training), is_training)

    def backward_pass(self, cached_input_signal, cached_pre_act, backward_signal):
        input_signal, x_m, mean, var = cached_pre_act

        delta = backward_signal

        d_x_hat = self.gamma * delta
        d_var = -0.5 * ((var + self.epsilon) ** 1.5) * np.sum(d_x_hat * x_m, axis=0)
        d_mean = -np.sum(d_x_hat, axis=0) / np.sqrt(var + self.epsilon) - 2 * d_var * np.mean(x_m, axis=0)
        next_backward_signal = d_x_hat / np.sqrt(var + self.epsilon) + 2 / len(
            input_signal) * d_var * x_m + d_mean / len(input_signal)
        grad_ = np.sum(delta, axis=0)
        grad = np.sum(delta * x_m / np.sqrt(var + self.epsilon), axis=0) + self.regularization.derivative(
            self.get_regularization_params())

        return next_backward_signal, grad_, grad

    def update(self, lr, step_grad_, step_grad):
        if self.is_trainable:
            self.gamma -= lr * step_grad
            self.betta -= lr * step_grad_

    def get_regularization_params(self):
        return self.gamma


class Flatten(Layer):

    def __str__(self):
        return 'Flatten'

    def __init__(self, input_shape: tuple = None, connected_to: Layer = None):
        super(Flatten).__init__()
        self.regularization = NoRegularization()

        if input_shape:
            self.input_shape = input_shape
        elif connected_to:
            self.input_shape = connected_to.output_shape
        else:
            raise ValueError('argument input_shape or connected_to must be not None')

        size = 1
        for s in self.input_shape[1:]:
            size *= s
        self.output_shape = (None, size)

    def pre_activate(self, input_signal, is_training=False):
        if is_training:
            return input_signal.reshape([len(input_signal), self.output_shape[1]]), input_signal.shape[1:]
        else:
            return input_signal.reshape([len(input_signal), self.output_shape[1]])

    def activate(self, pre_act, is_training=False):
        if is_training:
            return pre_act[0]
        else:
            return pre_act

    def forward_pass(self, input_signal, is_training=False):
        return self.activate(self.pre_activate(input_signal, is_training), is_training)

    def backward_pass(self, cached_input_signal, cached_pre_act, backward_signal):
        delta = backward_signal
        shape = cached_pre_act[1]
        next_backward_signal = delta.reshape(delta.shape[0], shape[0], shape[1], shape[2])
        grad_ = 0
        grad = 0
        return next_backward_signal, grad_, grad

    def update(self, lr, step_grad_, step_grad):
        pass

    def get_regularization_params(self):
        return 0


class Convolution2D(Layer):

    def __str__(self):
        return 'Convolution2D'

    def __init__(self, filter_shape, depth, stride, padding, activation, regularization=None, input_shape: tuple = None,
                 connected_to: Layer = None, weights: np.ndarray = None, bias: np.ndarray = None):
        super(Convolution2D).__init__()
        if input_shape is None and connected_to is None:
            raise ValueError('argument input_shape or connected_to must be not None')
        if input_shape:
            if len(input_shape) == 4:
                self.input_shape = input_shape
            else:
                raise ValueError('input_shape must have 4 dimensions.')
        if connected_to:
            if len(connected_to.output_shape) == 4:
                self.input_shape = connected_to.output_shape
            else:
                raise ValueError('output_shape of connected_to layer must have 4 dimensions.')

        if (self.input_shape[2] + 2 * padding - filter_shape[0]) % stride != 0:
            raise ValueError(
                'Incorrect size: ({} + 2*{} - {})%{} != 0'.format(self.input_shape[2], padding, filter_shape[0],
                                                                  stride))
        if (self.input_shape[3] + 2 * padding - filter_shape[1]) % stride != 0:
            raise ValueError(
                'Incorrect size: ({} + 2*{} - {})%{} != 0'.format(self.input_shape[3], padding, filter_shape[1],
                                                                  stride))

        if isinstance(activation, SoftMax):
            raise ValueError("Can't tuple {} as activation of Convolution2D layer.".format(activation))

        self.filter_shape = filter_shape  # [rows, columns]
        self.stride = stride
        self.padding = padding
        self.output_shape = (None, depth,
                             1 + (self.input_shape[2] + 2 * padding - filter_shape[0]) // stride,
                             1 + (self.input_shape[3] + 2 * padding - filter_shape[1]) // stride)
        self.activation = activation
        self.regularization = regularization

        if self.regularization is None:
            self.regularization = NoRegularization()

        weights_shape = (self.output_shape[1],
                         self.input_shape[1],
                         self.filter_shape[0],
                         self.filter_shape[1])

        if bias is None:
            self.bias = 1 / np.sqrt(self.input_shape[2] * self.input_shape[3] / 2) * np.random.rand(
                *(self.output_shape[1], 1))
        elif bias.shape == (self.output_shape[1], 1):
            self.bias = np.copy(bias)

        if weights is None:
            self.weights = 1 / np.sqrt(self.input_shape[1] * self.input_shape[2] * self.input_shape[3] / 2) * \
                           np.random.normal(loc=0,
                                            scale=1,
                                            size=weights_shape)
        elif weights.shape == weights_shape:
            self.weights = np.copy(weights)

    def pre_activate(self, input_signal, is_training=False):
        imscol = self.ims2col(input_signal, self.output_shape, self.filter_shape, self.stride, self.padding)
        wscol = self.weights.reshape(self.weights.shape[0], -1)
        d_prod = []
        for i in range(len(input_signal)):
            d_prod.append(np.dot(wscol, imscol[i]) + self.bias)
        d_prod = np.array(d_prod)
        d_prod = d_prod.reshape([len(d_prod), self.output_shape[1], self.output_shape[2], self.output_shape[3]])
        return d_prod

    def activate(self, pre_act, is_training=False):
        return self.activation.calculate(pre_act)

    def forward_pass(self, input_signal, is_training=False):
        return self.activate(self.pre_activate(input_signal, is_training), is_training)

    def backward_pass(self, cached_input_signal, cached_pre_act, backward_signal):
        delta = backward_signal * self.activation.derivative(cached_pre_act[0])
        dcol = delta.reshape([len(delta), self.output_shape[1], self.output_shape[2] * self.output_shape[3]])
        wscol = self.weights.reshape(self.weights.shape[0], -1)
        next_backward_signal_cols = []
        for i in range(len(backward_signal)):
            next_backward_signal_cols.append(np.dot(wscol.T, dcol[i]))
        next_backward_signal_cols = np.array(next_backward_signal_cols)
        next_backward_signal = self.cols2im(next_backward_signal_cols, self.input_shape, self.output_shape,
                                            self.filter_shape,
                                            self.stride,
                                            self.padding)

        imscols = self.ims2col(cached_input_signal, self.output_shape, self.filter_shape, self.stride, self.padding)
        dcols = []
        for d in delta:
            dcols.append(d.reshape(d.shape[0], -1))
        dcols = np.array(dcols)
        filter_shape = self.filter_shape
        w_grad_shape = (self.output_shape[1], self.input_shape[1] * filter_shape[0] * filter_shape[1])
        grad = np.zeros(shape=(w_grad_shape))
        grad_ = np.zeros(shape=(self.output_shape[1], 1))
        for k in range(dcols.shape[0]):
            grad += np.dot(dcols[k], imscols[k].T)
            grad_ += np.sum(dcols[k], axis=1, keepdims=True)
        grad = grad.reshape([self.output_shape[1], self.input_shape[1], filter_shape[0], filter_shape[1]])
        grad += self.regularization.derivative(self.get_regularization_params())

        return next_backward_signal, grad_, grad

    def update(self, lr, step_grad_, step_grad):
        self.weights -= lr * step_grad
        self.bias -= lr * step_grad_

    def get_regularization_params(self):
        return self.weights

    def pad_ims(self, im_arr, padding):
        return np.lib.pad(im_arr, ((0, 0), (0, 0), (padding, padding), (padding, padding)), 'constant',
                          constant_values=0)

    def im2col(self, im, output_shape, filter_shape, s):
        i_c, i_h, i_w = im.shape
        f_h, f_w = filter_shape
        o_h, o_w = output_shape[2:]

        col = np.zeros(shape=(i_c * f_h * f_w, o_h * o_w))
        for h in range(0, o_h, s):
            r = h * o_h
            for w in range(0, o_w, s):
                col[:, r + w] = im[:, h:h + f_h, w:w + f_w].flat
                # col[:, r + w] = im[:, h:h + f_h, w:w + f_w].flatten()
        return col

    def ims2col(self, ims, output_shape, filter_shape, stride, padding):
        p_ims = self.pad_ims(ims, padding)
        cols = np.empty(
            shape=(ims.shape[0], ims.shape[1] * filter_shape[0] * filter_shape[1], output_shape[2] * output_shape[3]))
        for i in range(ims.shape[0]):
            cols[i] = self.im2col(p_ims[i, :, :, :], output_shape, filter_shape, stride)
        return cols

    def col2im(self, col, input_shape, output_shape, filter_shape, s, p):
        f_h, f_w = filter_shape
        i_c, i_h, i_w = input_shape[1:]
        o_c, o_h, o_w = output_shape[1:]
        i_h += 2 * p
        i_w += 2 * p
        im = np.zeros(shape=(i_c, i_h, i_w))
        rs = [i for i in range(0, input_shape[2], s)]
        cs = [i for i in range(0, input_shape[3], s)]
        for k in range(col.shape[1]):
            i = k // o_h
            j = k % o_w
            im[:, rs[i]:rs[i] + f_h, cs[j]:cs[j] + f_w] += col[:, k].reshape([i_c, f_h, f_w])
        return im

    def cols2im(self, cols, input_shape, output_shape, filter_shape, stride, padding):
        ims = np.empty((cols.shape[0], input_shape[1], input_shape[2] + 2 * padding, input_shape[3] + 2 * padding))
        for i in range(cols.shape[0]):
            ims[i] = self.col2im(cols[i], input_shape, output_shape, filter_shape, stride, padding)
        if padding == 0:
            return ims
        else:
            return ims[:, :, padding:-padding, padding:-padding]


class MaxPool2D(Layer):

    def __str__(self):
        return "MaxPool2D"

    def __init__(self, window_shape, stride, input_shape: tuple = None, connected_to: Layer = None):
        super(MaxPool2D).__init__()
        if input_shape is None and connected_to is None:
            raise ValueError('argument input_shape or connected_to must be not None')
        if input_shape:
            if len(input_shape) == 4:
                self.input_shape = input_shape
            else:
                raise ValueError('input_shape must have 4 dimensions.')
        if connected_to:
            if len(connected_to.output_shape) == 4:
                self.input_shape = connected_to.output_shape
            else:
                raise ValueError('output_shape of connected_to layer must have 4 dimensions.')

        if (self.input_shape[2] - window_shape[0]) % stride != 0:
            raise ValueError('Incorrect size: ({} - {})%{} != 0'.format(self.input_shape[2], window_shape[0], stride))
        if (self.input_shape[3] - window_shape[1]) % stride != 0:
            raise ValueError('Incorrect size: ({} - {})%{} != 0'.format(self.input_shape[3], window_shape[1], stride))

        self.window_shape = window_shape
        self.stride = stride
        self.output_shape = (None, self.input_shape[1],
                             1 + (self.input_shape[2] - window_shape[0]) // stride,
                             1 + (self.input_shape[3] - window_shape[1]) // stride)
        self.regularization = NoRegularization()

    def pre_activate(self, input_signal, is_training=False):
        dot_prod = []
        pooled_ids = []
        for i in range(input_signal.shape[0]):
            colbych = self.im2colbychannel(im=input_signal[i],
                                           output_shape=self.output_shape,
                                           window_shape=self.window_shape,
                                           s=self.stride)
            dot_prod.append(
                colbych.max(axis=1, keepdims=True).reshape(colbych.shape[0], self.output_shape[2],
                                                           self.output_shape[3]))
            pooled_ids.append(colbych.argmax(axis=1))
        if is_training:
            return np.array(dot_prod), np.array(pooled_ids)
        else:
            return np.array(dot_prod)

    def activate(self, pre_act, is_training=False):
        if is_training:
            return pre_act[0]
        else:
            return pre_act

    def forward_pass(self, input_signal, is_training=False):
        return self.activate(self.pre_activate(input_signal, is_training), is_training)

    def backward_pass(self, cached_input_signal, cached_pre_act, backward_signal):
        pooled_ids = cached_pre_act[1]
        next_backward_signal = np.zeros(
            shape=(len(backward_signal), self.input_shape[1], self.input_shape[2], self.input_shape[3]))
        for i in range(backward_signal.shape[0]):
            for c in range(self.output_shape[1]):
                for o_h in range(self.output_shape[2]):
                    for o_w in range(self.output_shape[3]):
                        id = pooled_ids[i, c, o_h * self.output_shape[2] + o_w]
                        row = id // self.window_shape[0]
                        col = id % self.window_shape[1]
                        next_backward_signal[i, c, o_h + row, o_w + col] += backward_signal[i, c, o_h, o_w]
        grad_ = 0
        grad = 0
        return next_backward_signal, grad_, grad

    def update(self, lr, step_grad_, step_grad):
        pass

    def get_regularization_params(self):
        return 0

    def im2colbychannel(self, im, output_shape, window_shape, s):
        i_c, i_h, i_w = im.shape
        w_h, w_w = window_shape
        o_h, o_w = output_shape[2:]

        stretched_im = np.zeros(shape=(i_c, w_h * w_w, o_h * o_w))
        for h in range(0, o_h):
            row = h * s
            for w in range(0, o_w):
                col = w * s
                stretched_im[:, :, h * o_h + w] = im[:, row:row + w_h, col:col + w_w].reshape([i_c, w_h * w_w])
        return stretched_im


if __name__ == "__main__":
    pass
