import numpy as np
from keras.datasets import mnist

from NN.helper import LRScheduler, EarlyStopper, ModelSaver, calculate_accuracy
import NN.models as NN
import NN.regularizations as regularizations
import NN.layers as layers
import NN.activations as activations
import NN.losses as losses
import NN.optimizers as optimizers

num_classes = 10

(x_train, y_train), (x_test, y_test) = mnist.load_data()

img_h = x_train.shape[1]
img_w = x_train.shape[2]
x_train = x_train.reshape(x_train.shape[0], img_w * img_h).astype('float32') / 255
x_test = x_test.reshape(x_test.shape[0], img_w * img_h).astype('float32') / 255
y_train = np.array([np.array(np.eye(M=num_classes, N=1, k=int(d)).flat) for d in y_train])
y_test = np.array([np.array(np.eye(M=num_classes, N=1, k=int(d)).flat) for d in y_test])

print('x_train.shape', x_train.shape)
print('y_train.shape', y_train.shape)
print('x_test.shape', x_test.shape)
print('y_test.shape', y_test.shape)

nn = NN.NeuralNetwork(classification=True)
# saved_nn_file = None
saved_nn_file = './mnist_models/mlp_21.pickle'

regularization = regularizations.L2Regularization(1e-10)

if saved_nn_file:
    nn.load(saved_nn_file)
else:
    nn.push_layer(layers.FullyConnected(input_shape=(None, img_w * img_h), outputs=400, activation=activations.ArcTan(),
                                        regularization=regularization))
    nn.push_layer(layers.BatchNormalization(connected_to=nn.layers[-1]))
    nn.push_layer(layers.Dropout(prob=0.2, connected_to=nn.layers[-1]))

    nn.push_layer(layers.FullyConnected(outputs=400, activation=activations.ArcTan(),
                                        regularization=regularization, connected_to=nn.layers[-1]))
    nn.push_layer(layers.BatchNormalization(connected_to=nn.layers[-1]))
    nn.push_layer(layers.Dropout(prob=0.2, connected_to=nn.layers[-1]))

    nn.push_layer(layers.FullyConnected(outputs=num_classes, activation=activations.SoftMax(),
                                        regularization=regularization, connected_to=nn.layers[-1]))

print(nn)

pred_test = nn.forward_pass(x_test)
test_acc = calculate_accuracy(prediction=pred_test, target=y_test, one_hot_encoding=True, classification=True)
print('Accuracy on the test set:', test_acc)

max_epochs = 35
l_rate = 1e-3

loss_f = losses.CrossEntropy()
lr_scheduler = LRScheduler(l_rate, schedule={10: 5e-4, 20: 1e-5, 25: 5e-5, 30: 1e-5})
optimizer = optimizers.SGD(nn.layers,
                           lr_scheduler=lr_scheduler,
                           loss_function=loss_f)

train_loss_list, train_accuracy_list, valid_loss_list, valid_accuracy_list = nn.fit(
    train_set=(x_train, y_train),
    valid_set=(x_test, y_test),
    batch_size=128,
    max_epochs=max_epochs,
    optimizer=optimizer,
    model_saver=ModelSaver(model_name='mlp', folder_name='mnist_models', save_best=True, check_for='train_acc'),
    report=1,
    batch_report=0)
