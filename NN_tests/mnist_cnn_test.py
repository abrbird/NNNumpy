import numpy as np
from keras.datasets import mnist

from NN.helper import calculate_accuracy, ModelSaver, LRScheduler
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
img_ch = 1
x_train = x_train.reshape(x_train.shape[0], img_ch, img_h, img_w).astype('float32') / 255
x_test = x_test.reshape(x_test.shape[0], img_ch, img_h, img_w).astype('float32') / 255
y_train = np.array([np.array(np.eye(M=num_classes, N=1, k=int(d)).flat) for d in y_train])
y_test = np.array([np.array(np.eye(M=num_classes, N=1, k=int(d)).flat) for d in y_test])

print('x_train.shape', x_train.shape)
print('y_train.shape', y_train.shape)
print('x_test.shape', x_test.shape)
print('y_test.shape', y_test.shape)

nn = NN.NeuralNetwork(classification=True)
# saved_nn_file = None
saved_nn_file = './mnist_models/cnn_5.pickle'

# regularization = regularizations.L2Regularization(1e-10)
regularization = regularizations.NoRegularization()

if saved_nn_file:
    nn.load(saved_nn_file)
else:
    nn.push_layer(layers.Convolution2D(input_shape=(None, img_ch, img_h, img_w), filter_shape=(3, 3),
                                       activation=activations.ReLU(), stride=1, padding=1, depth=12))
    nn.push_layer(layers.MaxPool2D(window_shape=(2, 2), stride=2, connected_to=nn.layers[-1]))

    nn.push_layer(layers.Convolution2D(filter_shape=(3, 3), activation=activations.ReLU(), stride=1,
                                       padding=1, depth=24, connected_to=nn.layers[-1]))
    nn.push_layer(layers.MaxPool2D(window_shape=(2, 2), stride=2, connected_to=nn.layers[-1]))

    nn.push_layer(layers.Convolution2D(filter_shape=(3, 3), activation=activations.ReLU(), stride=1,
                                       padding=1, depth=48, connected_to=nn.layers[-1]))

    nn.push_layer(layers.Flatten(connected_to=nn.layers[-1]))
    nn.push_layer(layers.BatchNormalization(connected_to=nn.layers[-1]))
    nn.push_layer(
        layers.FullyConnected(outputs=nn.layers[-1].output_shape[-1] // 2, activation=activations.ReLU(),
                              regularization=regularization, connected_to=nn.layers[-1]))
    nn.push_layer(
        layers.FullyConnected(outputs=num_classes, activation=activations.SoftMax(),
                              regularization=regularization, connected_to=nn.layers[-1]))

print(nn)

max_epochs = 5
l_rate = 1e-5

loss_f = losses.CrossEntropy()
optimizer = optimizers.Adam(nn.layers, lr_scheduler=LRScheduler(l_rate), loss_function=loss_f)

pred_test = nn.forward_pass(x_test, report=1)
test_acc = calculate_accuracy(prediction=pred_test, target=y_test, one_hot_encoding=True, classification=True)
print('Accuracy on the test set:', test_acc)

train_loss_list, train_accuracy_list, valid_loss_list, valid_accuracy_list = nn.fit(
    train_set=(x_train, y_train),
    valid_set=(x_test, y_test),
    batch_size=500,
    max_epochs=max_epochs,
    optimizer=optimizer,
    model_saver=ModelSaver(model_name='cnn', folder_name='mnist_models', save_best=True),
    report=1,
    batch_report=1)
