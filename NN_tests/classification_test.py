import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import NN.models as NN
import NN.regularizations as regularizations
import NN.layers as layers
import NN.activations as activations
import NN.losses as losses
import NN.optimizers as optimizers
from NN.helper import LRScheduler, EarlyStopper


def dist(x, y):
    s = x - y
    return np.sqrt(np.dot(s, s))


n = 500
train_n = int(n * 0.75)
xor_generate = False
one_hot_encoding = False

ds_inputs = np.random.random_sample(size=(n, 2))


if xor_generate:
    ds = np.array(
        [np.array([el[0], el[1], 0]) if (el[0] < 0.5 and el[1] < 0.5) or (el[0] >= 0.5 and el[1] >= 0.5) else np.array(
            [el[0], el[1], 1]) for el in ds_inputs])
else:
    c = np.array([0.5, 0.5])
    rad = 0.3
    ds = np.array(
        [np.array([el[0], el[1], 0]) if dist(el[:2], c) < rad else np.array(
            [el[0], el[1], 1]) for el in ds_inputs])

train_ds = ds[:train_n]
test_ds = ds[train_n:]

train_x = train_ds[:, :-1]
train_y = train_ds[:, -1:]

test_x = test_ds[:, :-1]
test_y = test_ds[:, -1:]

if one_hot_encoding:
    train_y = np.array([np.array(np.eye(M=2, N=1, k=int(d)).flat) for d in train_y])
    test_y = np.array([np.array(np.eye(M=2, N=1, k=int(d)).flat) for d in test_y])

pd_ds = pd.DataFrame(ds)
zeros = pd_ds[pd_ds[2] == 0]
ones = pd_ds[pd_ds[2] == 1]

plt.plot(zeros[0], zeros[1], 'bo', label="0")
plt.plot(ones[0], ones[1], 'ro', label="1")

plt.xlabel('X1')
plt.ylabel('X2')
plt.ylim((-0.1, 1.3))
plt.xlim((-0.05, 1.05))
plt.title("Dataset")
legend = plt.legend(loc='upper center', shadow=True, fontsize='x-large')
legend.get_frame().set_facecolor('#00FFCC')

plt.show()

l_rate = 1e-4
max_iters = 8000
eps = 1e-9

if one_hot_encoding:
    nn_shape = (6, 2)
else:
    nn_shape = (6, 1)

regularization = regularizations.L1L2Regularization(l1_multiplier=1e-6, l2_multiplier=1e-6)
nn = NN.NeuralNetwork(classification=True, one_hot_encoding=one_hot_encoding)

nn.push_layer(layers.FullyConnected(nn_shape[0], activations.ArcTan(), regularization=regularization, input_shape=(None, 2)))
nn.push_layer(layers.BatchNormalization(connected_to=nn.layers[-1]))
nn.push_layer(
    layers.FullyConnected(nn_shape[1], activations.Sigmoid(), regularization=regularization, connected_to=nn.layers[-1]))

print(nn)

squared_error = losses.CrossEntropy()
optimizer = optimizers.Adam(nn.layers, lr_scheduler=LRScheduler(l_rate), loss_function=squared_error)

train_loss_list, train_accuracy_list, valid_loss_list, valid_accuracy_list = nn.fit(
    train_set=(train_x, train_y),
    valid_set=(test_x, test_y),
    batch_size=30,
    max_epochs=max_iters,
    optimizer=optimizer,
    early_stopper=EarlyStopper(eps),
    report=200)

print('epoch: {}, loss: {}, acc: {}, val_loss: {}, val_acc: {}'.format(len(train_loss_list),
                                                                       train_loss_list[-1],
                                                                       train_accuracy_list[-1],
                                                                       valid_loss_list[-1],
                                                                       valid_accuracy_list[-1]))

epochs = [i for i in range(len(train_loss_list))]
plt.plot(epochs, train_loss_list)
plt.xlabel('epochs')
plt.ylabel('Loss function')
plt.show()


input_signal = []
step = 0.05
rows_n = int((1 - 0) / step)
for i in range(0, rows_n + 1):
    for j in range(0, rows_n + 1):
        input_signal.append(np.array([step * i, step * j]))

input_signal = np.array(input_signal)
if one_hot_encoding:
    output_signal = np.array([np.argmax(r) for r in nn.forward_pass(input_signal)])
else:
    output_signal = nn.forward_pass(input_signal)

test_results = pd.DataFrame(input_signal)
test_results[2] = pd.Series(output_signal.flat)

if one_hot_encoding:
    zeros = test_results[test_results[2] == 0]
    ones = test_results[test_results[2] == 1]
else:
    zeros = test_results[test_results[2] < 0.5]
    ones = test_results[test_results[2] >= 0.5]

plt.clf()
plt.plot(zeros[0], zeros[1], 'bo', label="0")
plt.plot(ones[0], ones[1], 'ro', label="1")

plt.xlabel('X1')
plt.ylabel('X2')
plt.ylim((-0.1, 1.3))
plt.xlim((-0.05, 1.05))

legend = plt.legend(loc='upper center', shadow=True, fontsize='x-large')
legend.get_frame().set_facecolor('#00FFCC')
plt.show()
