import numpy as np
from keras.datasets import cifar10

import NN.models as NN
import NN.regularizations as regularizations
import NN.layers as layers
import NN.activations as activations
import NN.losses as losses
import NN.optimizers as optimizers
from NN.helper import LRScheduler, ModelSaver, calculate_accuracy

num_classes = 10

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

x_train = np.transpose(x_train, (0, 3, 1, 2))
x_test = np.transpose(x_test, (0, 3, 1, 2))
img_ch = x_train.shape[1]
img_h = x_train.shape[2]
img_w = x_train.shape[3]
x_train = x_train.reshape(x_train.shape[0], img_ch*img_h*img_w) / 255
x_test= x_test.reshape(x_test.shape[0], img_ch*img_h*img_w) / 255
y_train = np.array([np.array(np.eye(M=num_classes, N=1, k=int(d)).flat) for d in y_train])
y_test = np.array([np.array(np.eye(M=num_classes, N=1, k=int(d)).flat) for d in y_test])

print('x_train.shape', x_train.shape)
print('y_train.shape', y_train.shape)
print('x_test.shape', x_test.shape)
print('y_test.shape', y_test.shape)

regularization = regularizations.NoRegularization()
# regularization = regularizations.L1Regularization(1e-10)
# regularization = regularizations.L2Regularization(1e-6)
# regularization = regularizations.L1L2Regularization(1e-10, 1e-6)

# nn_data_pickle = None
nn_data_pickle = './cifar_10_models/mlp_10.pickle'

nn = NN.NeuralNetwork(classification=True)
if nn_data_pickle:
    nn.load(nn_data_pickle)
else:
    nn.push_layer(layers.FullyConnected(input_shape=(None, img_ch*img_h*img_w), outputs=350,
                                        activation=activations.ArcTan(), regularization=regularization))
    nn.push_layer(layers.BatchNormalization(connected_to=nn.layers[-1]))
    nn.push_layer(layers.FullyConnected(outputs=300, activation=activations.ArcTan(),
                                        regularization=regularization, connected_to=nn.layers[-1]))
    nn.push_layer(layers.BatchNormalization(connected_to=nn.layers[-1]))
    nn.push_layer(layers.FullyConnected(outputs=200, activation=activations.ArcTan(),
                                        regularization=regularization, connected_to=nn.layers[-1]))

    nn.push_layer(layers.FullyConnected(outputs=num_classes, activation=activations.SoftMax(),
                                        regularization=regularization, connected_to=nn.layers[-1]))
print(nn)

l_rate = 1e-3
max_epochs = 10

loss_f = losses.CrossEntropy()
optimizer = optimizers.Adam(nn.layers,
                            lr_scheduler=LRScheduler(l_rate,
                                                     schedule={3: 7.5e-4, 5: 5e-4, 7: 2.5e-4, 9: 1e-4, 12: 5e-5}),
                            loss_function=loss_f)

pred_test = nn.forward_pass(x_test, report=1)
test_acc = calculate_accuracy(prediction=pred_test, target=y_test, one_hot_encoding=True, classification=True)
print('Accuracy on the test set:', test_acc)

train_loss_list, train_accuracy_list, valid_loss_list, valid_accuracy_list = nn.fit(
    train_set=(x_train, y_train),
    valid_set=(x_test, y_test),
    batch_size=200,
    max_epochs=max_epochs,
    optimizer=optimizer,
    model_saver=ModelSaver('mlp', folder_name='cifar_10_models', save_best=True, check_for='valid_acc'),
    report=1,
    batch_report=10)

