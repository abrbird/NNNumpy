import numpy as np
from keras.datasets import cifar10

import NN.losses as losses
import NN.activations as activations
import NN.optimizers as optimizers
from NN.models import NeuralNetwork
from NN.layers import FullyConnected, Convolution2D, MaxPool2D, Flatten, BatchNormalization, Activation
from NN.regularizations import NoRegularization, L1Regularization, L2Regularization, L1L2Regularization

from NN.helper import calculate_accuracy, LRScheduler, ModelSaver

num_classes = 10

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

x_train = np.transpose(x_train, (0, 3, 1, 2)) / 255
x_test = np.transpose(x_test, (0, 3, 1, 2)) / 255
img_ch = x_train.shape[1]
img_h = x_train.shape[2]
img_w = x_train.shape[3]
y_train = np.array([np.array(np.eye(M=num_classes, N=1, k=int(d)).flat) for d in y_train])
y_test = np.array([np.array(np.eye(M=num_classes, N=1, k=int(d)).flat) for d in y_test])

print('x_train.shape', x_train.shape)
print('y_train.shape', y_train.shape)
print('x_test.shape', x_test.shape)
print('y_test.shape', y_test.shape)

# regularization = NoRegularization()
# regularization = L1Regularization(1e-12)
regularization = L2Regularization(1e-6)
# regularization = L1L2Regularization(1e-12, 1e-6)

# nn_data_pickle = None
nn_data_pickle = './cifar_10_models/cnn_12.pickle'

nn = NeuralNetwork(classification=True)
if nn_data_pickle:
    nn.load(nn_data_pickle)
else:
    bn_trainable = True
    nn.push_layer(Convolution2D(input_shape=(None, img_ch, img_h, img_w), filter_shape=(3, 3),
                                activation=activations.Linear(), stride=1, padding=1, depth=12))
    nn.push_layer(BatchNormalization(connected_to=nn.layers[-1]))
    nn.push_layer(Activation(activations.ReLU(), connected_to=nn.layers[-1]))

    nn.push_layer(Convolution2D(filter_shape=(3, 3), activation=activations.Linear(),
                                stride=1, padding=1, depth=12, connected_to=nn.layers[-1]))
    nn.push_layer(BatchNormalization(connected_to=nn.layers[-1]))
    nn.push_layer(Activation(activations.ReLU(), connected_to=nn.layers[-1]))

    nn.push_layer(MaxPool2D(window_shape=(2, 2), stride=2, connected_to=nn.layers[-1]))

    nn.push_layer(Convolution2D(filter_shape=(3, 3), activation=activations.Linear(),
                                stride=1, padding=1, depth=24, connected_to=nn.layers[-1]))
    nn.push_layer(BatchNormalization(connected_to=nn.layers[-1]))
    nn.push_layer(Activation(activations.ReLU(), connected_to=nn.layers[-1]))

    nn.push_layer(Convolution2D(filter_shape=(3, 3), activation=activations.Linear(),
                                stride=1, padding=1, depth=24, connected_to=nn.layers[-1]))
    nn.push_layer(BatchNormalization(connected_to=nn.layers[-1]))
    nn.push_layer(Activation(activations.ReLU(), connected_to=nn.layers[-1]))

    nn.push_layer(MaxPool2D(window_shape=(2, 2), stride=2, connected_to=nn.layers[-1]))

    nn.push_layer(Flatten(connected_to=nn.layers[-1]))
    nn.push_layer(BatchNormalization(connected_to=nn.layers[-1]))
    nn.push_layer(FullyConnected(outputs=nn.layers[-2].output_shape[1] // 4, activation=activations.ReLU(),
                                 regularization=regularization, connected_to=nn.layers[-1]))
    nn.push_layer(FullyConnected(outputs=num_classes, activation=activations.SoftMax(),
                                 regularization=regularization, connected_to=nn.layers[-1]))

print(nn)

l_rate = 1e-5
max_epochs = 12

loss_f = losses.CrossEntropy()

sched = {2: 7e-4, 3: 4e-4, 4: 1e-4, 5: 7.5e-5, 6: 3e-5, 7: 2e-5, 8: 1e-5}
# lr_scheduler=LRScheduler(l_rate, schedule=sched)
lr_scheduler = LRScheduler(l_rate)
optimizer = optimizers.Adam(nn.layers, lr_scheduler=lr_scheduler, loss_function=loss_f)

pred_test = nn.forward_pass(x_test, report=1)
test_acc = calculate_accuracy(prediction=pred_test, target=y_test, one_hot_encoding=True, classification=True)
print('Accuracy on the test set:', test_acc)

train_loss_list, train_accuracy_list, valid_loss_list, valid_accuracy_list = nn.fit(
    train_set=(x_train, y_train),
    valid_set=(x_test, y_test),
    batch_size=200,
    max_epochs=max_epochs,
    optimizer=optimizer,
    model_saver=ModelSaver(model_name='cnn', folder_name='cifar_10_models'),
    report=1,
    batch_report=1)

