import numpy as np
import pickle
from NN.helper import calculate_accuracy, get_batch_indexes, EarlyStopper, ModelSaver
from NN.optimizers import SGD

import NN.layers as lyrs


class NeuralNetwork:

    def __str__(self):
        result = ['NeuralNetwork ({} layers) input_shape: {}\n'.format(str(self.get_layers_num()),
                                                                       self.layers[0].input_shape if len(
                                                                           self.layers) > 0 else None)]
        total_params = 0
        if self.get_layers_num() > 0:
            result += ["# : Layer name (output_shape)\t: description\n"]
            for i, layer in enumerate(self.layers):
                total_params += layer.get_weights_count()
                result += ["{} : {} {}\t: {}\n".format(str(i + 1), str(layer), layer.output_shape, layer.description())]
        result += ["Total params num: {}".format(str(total_params))]
        return "".join(result)

    def __init__(self, classification, one_hot_encoding=None):
        self.layers = []
        self.classification = classification
        self.one_hot_encoding = one_hot_encoding

    def get_layers_num(self):
        return len(self.layers)

    def push_layer(self, layer: lyrs.Layer):
        self.layers.append(layer)

    def pop_layer(self):
        return self.layers.pop(-1)

    def forward_pass(self, input_signal, batch_size=1000, report=False):
        if input_signal.shape[1:] != self.layers[0].input_shape[1:]:
            print('Incorrect shape of input_signal')
        batch_size = np.min([batch_size, len(input_signal)])
        if batch_size <= 0:
            return None
        batch_indexes = get_batch_indexes(len(input_signal), batch_size)
        result_shape = [len(input_signal)] + [k for k in self.layers[-1].output_shape[1:]]
        result = np.empty(result_shape)
        if report and report > 0:
            print('Forward pass: ')
        for b, j in enumerate(batch_indexes):
            if report and report > 0 and (b + 1) % report == 0:
                print('{}/{}'.format(b + 1, len(batch_indexes)))
            signal = input_signal[j:j + batch_size]
            for layer in self.layers:
                signal = layer.forward_pass(signal, is_training=False)
            result[j:j + batch_size] = signal
        return result

    def forward_pass_cached(self, input_signal, is_training):
        """len of input_signal must be less than 1000"""
        if input_signal.shape[1:] != self.layers[0].input_shape[1:]:
            print('Incorrect shape of input_signal')
        if len(input_signal) < 1000 or len(input_signal.shape) == 2:
            pre_activations = []
            activation_signals = [input_signal]
            for layer in self.layers:
                pre_activations.append(layer.pre_activate(activation_signals[-1], is_training=is_training))
                activation_signals.append(layer.activate(pre_activations[-1], is_training=is_training))
            return pre_activations, activation_signals
        else:
            return None

    def backward(self, input_signals, pre_activations, backward_signal):
        deltas = []
        grads = []
        for i in range(len(self.layers) - 1, 0 - 1, -1):
            backward_signal, delta, grad = self.layers[i].backward_pass(input_signals[i], pre_activations[i],
                                                                        backward_signal)
            deltas.insert(0, delta)
            grads.insert(0, grad)
        return deltas, grads

    def fit(self,
            train_set,
            max_epochs: int,
            batch_size: int = 200,
            valid_set=None,
            optimizer=None,
            early_stopper=None,
            model_saver=None,
            report: int = 0,
            batch_report: int = 0):

        train_loss_list = []
        train_accuracy_list = []
        valid_loss_list = []
        valid_accuracy_list = []

        train_input, train_target = train_set

        valid_input = None
        valid_target = None
        validate = False
        if valid_set is not None and len(valid_set) > 0:
            valid_input, valid_target = valid_set
            validate = True

        if self.one_hot_encoding is None:
            self.one_hot_encoding = False if self.layers[-1].output_shape[-1] == 1 else True

        if optimizer is None:
            optimizer = SGD(self.layers)

        if early_stopper is None:
            early_stopper = EarlyStopper(eps=1e-10)

        if model_saver is None:
            model_saver = ModelSaver()

        batch_size = np.min([batch_size, len(train_input)])
        if batch_size <= 0:
            raise ValueError('batch_size must be positive (or train set has no elements)')
            # return train_loss_list, train_accuracy_list, valid_loss_list, valid_accuracy_list

        indexes = np.arange(0, len(train_input))
        # print(len(indexes))

        batch_indexes = get_batch_indexes(len(train_input), batch_size)

        epochs = 0
        training = epochs < max_epochs

        if report:
            print('Start fitting. Train set size: {}, valid set size: {}. Optimizer: {}, max_epochs: {}'.format(
                len(train_target), len(valid_target) if validate else None,
                str(optimizer), max_epochs))
        while training:
            epochs += 1

            if report and report > 0:
                if epochs % report == 0:
                    print('Start epoch: {}, learning_rate: {}'.format(epochs,
                                                                      optimizer.lr_scheduler.get_learning_rate(epochs)))
            for b, j in enumerate(batch_indexes):
                train_batch_inputs = train_input[indexes[j: j + batch_size]]
                train_batch_labels = train_target[indexes[j: j + batch_size]]

                pre_activations, activation_signals = self.forward_pass_cached(train_batch_inputs, is_training=True)
                loss_derivative = optimizer.loss_function.derivative(activation_signals[-1], train_batch_labels)

                deltas, grads = self.backward(activation_signals, pre_activations, loss_derivative)
                optimizer.optimize(epochs, deltas, grads)

                if batch_report and batch_report > 0 and (b + 1) % batch_report == 0:
                    batch_prediction = self.forward_pass(train_batch_inputs, batch_size=batch_size)

                    batch_loss = optimizer.loss_function.calculate(batch_prediction, train_batch_labels,
                                                                   self.one_hot_encoding)
                    for l in self.layers:
                        batch_loss += l.regularization.calculate(l.get_regularization_params())
                    batch_accuracy = calculate_accuracy(batch_prediction, train_batch_labels,
                                                        self.one_hot_encoding, self.classification)
                    print(
                        '\tbatch: {}/{}, loss: {}, acc: {}'.format(b + 1, len(batch_indexes),
                                                                   batch_loss,
                                                                   batch_accuracy))

            train_prediction = self.forward_pass(train_input, batch_size=500)

            train_loss = optimizer.loss_function.calculate(train_prediction, train_target, self.one_hot_encoding)
            for l in self.layers:
                train_loss += l.regularization.calculate(l.get_regularization_params())
            train_accuracy = calculate_accuracy(train_prediction, train_target,
                                                self.one_hot_encoding, self.classification)
            train_loss_list.append(train_loss)
            train_accuracy_list.append(train_accuracy)

            if validate:
                valid_prediction = self.forward_pass(valid_input, batch_size=500)
                valid_loss = optimizer.loss_function.calculate(valid_prediction, valid_target, self.one_hot_encoding)
                for l in self.layers:
                    valid_loss += l.regularization.calculate(l.get_regularization_params())
                valid_accuracy = calculate_accuracy(valid_prediction, valid_target,
                                                    self.one_hot_encoding, self.classification)
                valid_loss_list.append(valid_loss)
                valid_accuracy_list.append(valid_accuracy)
            else:
                valid_loss_list.append(None)
                valid_accuracy_list.append(None)

            training = epochs < max_epochs and not early_stopper.check(train_loss_list, train_accuracy_list,
                                                                       valid_loss_list, valid_accuracy_list)
            if report and report > 0:
                if epochs % report == 0 or not training:
                    print('End epoch: {}, loss: {}, acc: {}, val_loss: {}, val_acc: {}'.format(
                        epochs,
                        train_loss_list[-1],
                        train_accuracy_list[-1],
                        valid_loss_list[-1],
                        valid_accuracy_list[-1]))

            model_saver.save(self, epochs, train_loss_list, train_accuracy_list,
                             valid_loss_list, valid_accuracy_list, report)
            np.random.shuffle(batch_indexes)
        return train_loss_list, train_accuracy_list, valid_loss_list, valid_accuracy_list

    def save(self, filename):
        with open(filename, 'wb') as f:
            data = {'layers': [layer for layer in self.layers]}
            pickle.dump(data, f)

    def load(self, filename):
        with open(filename, 'rb') as f:
            data = pickle.load(f)
            self.layers = data['layers']


if __name__ == "__main__":
    pass
