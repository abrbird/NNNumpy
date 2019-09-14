import numpy as np
import matplotlib.pyplot as plt

from NN.models import NeuralNetwork
from NN.layers import FullyConnected, BatchNormalization
from NN.regularizations import NoRegularization, L1Regularization, L2Regularization, L1L2Regularization
from NN.optimizers import Adam, SGD
from NN.losses import SquaredError, AbsoluteError
from NN.helper import LRScheduler, EarlyStopper
import NN.activations as activations

num = 30

l_space = np.linspace(-2, 2.5, num=num).reshape((num, 1))
f1 = (l_space ** 3 + 5 * np.cos(l_space) + np.random.normal(loc=0, scale=0.25, size=l_space.shape))[:, -1]
f2 = (l_space ** 2 + 5 * np.sin(l_space) + np.random.normal(loc=0, scale=0.25, size=l_space.shape))[:, -1]
f = np.array([[x, y] for x, y in zip(f1, f2)])

train_x = l_space
train_y = f

plt.plot(l_space, f[:, 0], '.')
plt.plot(l_space, f[:, 1], '.')
plt.title('Train set')
plt.show()

reg_fs = [NoRegularization(), L2Regularization(l2_multiplier=3), L1L2Regularization(1, 2)]
loss_fs = [SquaredError(), AbsoluteError()]
reg_f = reg_fs[1]
loss_f = loss_fs[0]

nn = NeuralNetwork(classification=False, one_hot_encoding=False)
nn.push_layer(FullyConnected(input_shape=(None, 1), outputs=20, activation=activations.ArcTan(), regularization=reg_f))
nn.push_layer(BatchNormalization(connected_to=nn.layers[-1]))
nn.push_layer(FullyConnected(20, activations.Sinusoid(), regularization=reg_f, connected_to=nn.layers[-1]))
nn.push_layer(BatchNormalization(connected_to=nn.layers[-1]))
nn.push_layer(FullyConnected(15, activations.ArcTan(), regularization=reg_f, connected_to=nn.layers[-1]))
nn.push_layer(FullyConnected(2, activations.Linear(), regularization=reg_f, connected_to=nn.layers[-1]))

print(nn)

l_rate = 1e-3
max_epochs = 10000

lr_scheduler = LRScheduler(init_lr=l_rate, schedule={3000: 5e-4, 5000: 1e-4, 7000: 5e-5, 8000: 1e-5})

optimizer = Adam(nn.layers, lr_scheduler=lr_scheduler, loss_function=loss_f, opt_multipliers=(0.9, 0.95))
# optimizer = SGD(layers=nn.layers, lr_scheduler=lr_scheduler, loss_function=loss_f)

early_stopper = EarlyStopper(eps=1e-6)

train_loss_list, train_accuracy_list, valid_loss_list, valid_accuracy_list = nn.fit(
    train_set=(train_x, train_y),
    valid_set=None,
    batch_size=num,
    max_epochs=max_epochs,
    optimizer=optimizer,
    early_stopper=early_stopper,
    report=1000)

print('Epoch: {}, loss: {}, acc: {}, val_loss: {}, val_acc: {}'.format(len(train_loss_list),
                                                                       train_loss_list[-1],
                                                                       train_accuracy_list[-1],
                                                                       valid_loss_list[-1],
                                                                       valid_accuracy_list[-1]))

l_sp = np.linspace(-2.1, 2.6, num=100)
l_sp = np.array([[s] for s in l_sp])
line = nn.forward_pass(l_sp)
plt.plot(l_space, f[:, 0], '.')
plt.plot(l_space, f[:, 1], '.')
plt.plot(l_sp, line[:, 0], label='f1 - approximation')
plt.plot(l_sp, l_sp ** 3 + 5 * np.cos(l_sp), label='f1*')

plt.plot(l_sp, line[:, 1], label='f2 - approximation')
plt.plot(l_sp, l_sp ** 2 + 5 * np.sin(l_sp), label='f2*')
plt.legend()
plt.show()


epochs = [i for i in range(len(train_loss_list))]
plt.plot(epochs, train_loss_list)
plt.show()
