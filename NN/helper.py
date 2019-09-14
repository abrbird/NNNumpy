import numpy as np
import os


def calculate_accuracy(prediction, target, one_hot_encoding, classification):
    if one_hot_encoding:
        prediction = np.argmax(prediction, axis=1)
        target = np.argmax(target, axis=1)
    else:
        if classification:
            prediction = prediction > 0.5
    accuracy = np.mean(prediction == target)
    return accuracy


def get_batch_indexes(ds_size, batch_size):
    batch_size = np.min([batch_size, ds_size])
    iterations = int(np.ceil(ds_size / batch_size))
    batch_indexes = [j * batch_size for j in range(iterations)]
    return batch_indexes


class LRScheduler:

    def __init__(self, init_lr, schedule: dict = None):
        self.init_lr = init_lr
        self.schedule = schedule

    def get_learning_rate(self, epoch):
        if self.schedule is None:
            return self.init_lr
        else:
            lr = self.init_lr
            for k in self.schedule.keys():
                if epoch >= k:
                    lr = self.schedule.get(k)
            return lr


class EarlyStopper:

    def __init__(self, eps=1e-6):
        self.eps = eps

    def check(self, train_loss_list, train_accuracy_list, valid_loss_list, valid_accuracy_list):
        """checks plateau"""
        if len(train_loss_list) > 2 and abs(train_loss_list[-1] - train_loss_list[-2]) < self.eps:
            return True
        return False


class ModelSaver:
    check_for_list = ['train_loss',
                  'valid_loss',
                  'train_acc',
                  'valid_acc']

    def __init__(self, model_name=None, folder_name=None, save_best: bool = False, check_for=None):
        self.model_name = model_name
        self.folder_name = folder_name
        self.save_best = save_best
        self.check_for = check_for

        self.save_path = None
        self.last_loss = None
        self.last_acc = None

        if self.folder_name is None:
            self.save_path = os.getcwd()
        else:
            self.save_path = os.path.join(os.getcwd(), folder_name)

        if not os.path.isdir(self.save_path):
            os.makedirs(self.save_path)

        if self.check_for is None:
            self.check_for = ModelSaver.check_for_list[0]
        if self.check_for not in ModelSaver.check_for_list:
            self.check_for = ModelSaver.check_for_list[0]

    def save(self, model, epoch, train_loss_list, train_accuracy_list, valid_loss_list, valid_accuracy_list,
             report=False):
        if report and report > 0:
            report = True
        if model and self.model_name is not None:
            path = '{}.pickle'.format(os.path.join(self.save_path, '{}_{}'.format(self.model_name, epoch)))
            if self.save_best and epoch > 1:
                loss_list = None
                acc_list = None

                if self.check_for == ModelSaver.check_for_list[0]:
                    loss_list = train_loss_list
                if self.check_for == ModelSaver.check_for_list[1]:
                    loss_list = valid_loss_list
                if self.check_for == ModelSaver.check_for_list[2]:
                    acc_list = train_accuracy_list
                if self.check_for == ModelSaver.check_for_list[3]:
                    acc_list = valid_accuracy_list

                if loss_list:
                    if loss_list[-1] is None:
                        loss_list = train_loss_list
                    if epoch == 2:
                        self.last_loss = loss_list[-2]
                    if loss_list[-1] <= self.last_loss:
                        self.last_loss = loss_list[-1]
                        model.save(path)
                        if report:
                            print('Model saved.')
                    elif report:
                        print('Loss is not decreased.')

                if acc_list:
                    if acc_list[-1] is None:
                        acc_list = train_accuracy_list
                    if epoch == 2:
                        self.last_acc = acc_list[-2]
                    if acc_list[-1] >= self.last_acc:
                        self.last_acc = acc_list[-1]
                        model.save(path)
                        if report:
                            print('Model saved.')
                    elif report:
                        print('Accuracy is not improved.')

            else:
                model.save(path)
                if report:
                    print('Model saved.')
