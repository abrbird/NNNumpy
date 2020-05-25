import numpy as np
import matplotlib.pyplot as plt
# import time


# функции Ядра
# на вход подаются два вектора
# на выходе получаем меру расстояния между заданными векторами
# для нелинейных ядер можно задать константы, изменяюшие свойства ядра

def linear_kernel(x1, x2):
    '''Линейное ядро'''
    return np.dot(x1, x2)


def polynomial_kernel(x1, x2, k=1, const=0, d: int = 2):
    '''Полиномиальное ядро'''
    return (k * np.dot(x1, x2) + const) ** d


def sigmoid_kernel(x1, x2, k_0=0, k_1=0.5):
    '''Сигмоидальное ядро'''
    return 1. / (1. + np.exp(-1 * (k_0 + k_1 * np.dot(x1, x2))))


def tanh_kernel(x1, x2, k_0=0, k_1=0.1):
    '''Гиперболический тангенс'''
    return np.tanh(k_1 * np.dot(x1, x2) + k_0)


def rbf_kernel(x1, x2, sigma=0.2):
    '''Радиально-базисная функция ядра'''
    s = x1 - x2
    return np.exp(-1 * (np.dot(s, s) / (2 * sigma ** 2)))


def gauss_kernel(x1, x2, sigma=0.1):
    '''Гауссовское ядро'''
    s = x1 - x2
    return np.exp(-1 * np.dot(s, s) / sigma)


def laplace_kernel(x1, x2, sigma=0.05):
    '''Лапласовское ядро'''
    # при sigma = 0.1 модель переобучается
    s = x1 - x2
    return np.exp(-1 * np.sqrt(np.dot(s, s)) / sigma)


class MaxLossSVM(object):

    def __init__(self, C=1.0, lmbd=0.5, kernel=linear_kernel):
        """
        :param C: константа С перед штрафом за ошибку
        :param lmbd: константа лямбда в двойственной задаче
                    если lmbd ближе к 0, то альфа разрастаются
                    если lmbd ближе к 1, то альфы остаются маленькими
                    если задать lmлаbd = 1, то метод не сойдется
        :param kernel: функция ядра
        """
        self.C = C
        self.lmbd = lmbd

        # в питоне функции ведут себя как объекты, поэтому их можно передевать в качестве аргументов
        # а вызывать функции можно приписывая скобки: self.kernel(*args)
        self.kernel = kernel

        # массив меток обучающей выборки
        self.Y = None
        # массив точек (объектов) обучающей выборки
        self.X = None
        # массив точек обучающей выборки, дополненный столбцом из единиц
        self.X_extended = None
        # вектор для хранения значений альфа
        self.alpha = None

    def fit(self, train_data, test_data=None, eps=1e-3, max_iters: int = 150, info=False, plotting=None):
        """
        :param train_data: обучающая выборка
        :param test_data: тестовая выборка
        :param eps: значение для критерия остановки
        :param max_iters: максимальное количество итераций
        :param info: булевая переменная, если True - выводится информация на каждой итерации
        :param plotting: кортеж (period, left_lim, right_lim),
                period - периодичность вывода графиков (в количествах итераций)
                left_lim - левая граница графика
                right_lim - правая граница графика
        :return:
        """

        l = train_data.shape[0]  # количество точек в обучающей выборке
        n = train_data.shape[1] - 1  # размерность пространства точек (-1, потому что последнее значение - метка)

        # train_data при l=3 и n=4 имеет вид (test_data имеет похожий вид, только l может быть другим)
        # [
        # [x11, x12, x13, x14, y1],
        # [x21, x22, x23, x24, y2],
        # [x31, x32, x33, x34, y3],
        # ]

        # берем все строки и последний столбец
        self.Y = train_data[:, -1:]
        # берем все строки и все столбцы, кроме последнего столбца
        self.X = train_data[:, :-1]
        # дополняем матрицу точек единицами
        self.X_extended = self.__extend__(self.X)
        # инициализируем вектор альфа нулями
        self.alpha = np.zeros_like(self.Y)

        # разделим обучающую выборку на два класса для отрисовки графиков
        if plotting is not None:
            class_one = []
            class_two = []
            for d in train_data:
                if d[-1] == 1:
                    class_one.append(d)
                elif d[-1] == -1:
                    class_two.append(d)

        cost_list = []  # массив для сохранения значений целевой функции на обучающей выборке
        accuracy_list = []  # массив для сохранения точности классификации на обучающей выборке
        test_cost_list = []  # массив для сохранения значений целевой функции на тестовой выборке
        test_accuracy_list = []  # массив для сохранения точности классификации на тестовой выборке

        # Матрица А - матрица, состоящая из попарных применений ядерной функции к точкам из обучающей выборки
        A = self.create_A(self.X_extended, self.X_extended)

        if test_data is not None:
            # если задана тестовая выборка, посчитаем для нее матрицу попарных применений ядра,
            # чтобы не считать заново на каждой итерации
            X_T = test_data[:, :-1]
            Y_T = test_data[:, -1:]
            A_T = self.create_A(self.X_extended, self.__extend__(X_T))

        A_ = (A.T * self.Y).T * self.Y

        # находим антиградиент

        gs = [self.cost_f_d1(A, self.Y, self.alpha, self.lmbd)]
        gammas = [0]
        ps = [-gs[-1]]
        bettas = [-np.dot(ps[-1].T, gs[-1])/np.dot(np.dot(A_, ps[-1]).T, ps[-1])]

        self.alpha = self.alpha + bettas[-1]*ps[-1]

        # проецируем альфа на допустимое множество (избавляемся от отрицательных значений)
        self.alpha = np.clip(self.alpha, 0, a_max=None)

        stop_fitting = False

        while len(cost_list) < max_iters and not stop_fitting:
            # находим антиградиент
            gs.append(self.cost_f_d1(A, self.Y, self.alpha, self.lmbd))
            gammas.append(np.dot(gs[-1].T, np.dot(A_, ps[-1])) /np.dot(np.dot(A_, ps[-1]).T, ps[-1]))
            ps.append(gammas[-1]*ps[-1]-gs[-1])
            bettas.append(-np.dot(gs[-1].T, ps[-1])/np.dot(np.dot(A_, ps[-1]).T, ps[-1]))
            new_alpha = self.alpha+bettas[-1]*ps[-1]

            new_alpha = np.clip(new_alpha, 0, a_max=None)
            difference = np.abs(self.alpha - new_alpha)
            # если новый вектор слабо отличается от старого, можно ссчитать, что метод сошелся
            if np.mean(difference) < eps:
                stop_fitting = True
            # else:
            #     print(np.mean(difference))
            # обновляем вектор альфа
            self.alpha = new_alpha

            train_project = np.sum((A.T * self.Y * self.alpha).T, axis=0).reshape((l, 1))
            train_predict = np.sign(train_project)
            cost_list.append(0.5 * np.sum((A_.T * self.alpha).T * self.alpha)
                             + (self.lmbd - 1) * np.sum(self.alpha) - self.lmbd * self.C)
            accuracy_list.append(self.calc_acc(train_predict, self.Y))

            if test_data is not None:
                test_project = np.sum((A_T.T * (self.Y * self.alpha).T).T, axis=0).reshape((len(Y_T), 1))
                test_predict = np.sign(test_project)
                test_cost_list.append(None)
                test_accuracy_list.append(self.calc_acc(test_predict, Y_T))

            else:
                test_cost_list.append(None)
                test_accuracy_list.append(None)

            if info:
                if plotting is not None:
                    period, gr_l, gr_r = plotting
                    if len(cost_list) % period == 0:
                        self.plot(np.array(class_one), np.array(class_two), grid_l=gr_l, grid_r=gr_r)
                print('it: {}, cost: {}, acc: {}, val_cost: {}, val_acc: {}'.format(len(cost_list), cost_list[-1],
                                                                                    accuracy_list[-1],
                                                                                    test_cost_list[-1],
                                                                                    test_accuracy_list[-1]))
        else:
            if plotting is not None:
                period, gr_l, gr_r = plotting
                self.plot(np.array(class_one), np.array(class_two), grid_l=gr_l, grid_r=gr_r)
        return cost_list, accuracy_list, test_cost_list, test_accuracy_list

    def create_A(self, X1, X2):
        A = np.zeros(shape=(X1.shape[0], X2.shape[0]))
        for i in range(X1.shape[0]):
            for j in range(X2.shape[0]):
                A[i, j] = self.kernel(X1[i], X2[j])
        return A

    def bi_search(self, dir, A, Y, a=0, b=0.5, delta=1e-7, tol=1e-10, max_its=16):
        """
        :param dir: вектор направяления
        :param A:
        :param Y: метки
        :param a: начало отрезка, на котором ищется шаг
        :param b: конец отрезка, на котором ищется шаг
        :param delta: параметр для отступа от середины отрезка
        :param tol: параметр для критерия остановки
        :param max_its: максимальное количество итераций
        :return: длина шага
        """
        its = 0
        while True:
            cost_f_list = []
            middle = (a + b) / 2 # находим середину отрезка
            ls = [middle - delta, middle + delta] # вычисляем знаяения: середина - delta, середина + delta
            for l in ls:
                a_new = self.alpha + l * dir # вычисляем новый вектор альфа
                a_new = np.clip(a_new, 0, a_max=None) # избавляемся от отризацтельных значений
                cost_f_list.append(self.__cost_f__(A, Y, a_new, self.lmbd, self.C))
                # S = (A.T * (Y * a_new).T).T
                # train_project = np.sum(S, axis=0).reshape((len(Y), 1))
                # cost_f_list.append(0.5 * np.sum(train_project ** 2) + (self.lmbd - 1) * np.sum(a_new))
            if cost_f_list[0] < cost_f_list[1]:
                b = ls[1]
            else:
                a = ls[0]
            if its > max_its or abs(a - b) < tol:
                l = (a + b) / 2
                break
            its += 1
        return l

    def __cost_f__(self, A, Y, alpha, lmbd, C):
        A_ = (A.T * Y * alpha).T * Y * alpha
        return 0.5 * np.sum(A_) + (lmbd - 1) * np.sum(alpha) - lmbd * C

    def cost_f_d1(self, A, Y, alpha, lmbd):
        # первая производная
        A_ = (A.T * Y).T * Y
        # return 0.5 * np.dot(A_, alpha) +  np.dot(A.diagonal(), alpha) + (lmbd - 1)
        return 0.5 * np.dot(A_, alpha) + (lmbd - 1)

    def cost_f_d2(self, A, Y):
        # вторая производная
        A_ = (A.T * Y).T * Y
        return A_

    def plot(self, cl_1, cl_2, grid_l=0., grid_r=1.):
        if self.X.shape[1] == 2:
            plt.plot(cl_1[:, 0], cl_1[:, 1], "b+")
            plt.plot(cl_2[:, 0], cl_2[:, 1], "r+")
            x1, x2 = np.meshgrid(np.linspace(grid_l, grid_r, 50), np.linspace(grid_l, grid_r, 50))
            x = np.array([[x1, x2] for x1, x2 in zip(np.ravel(x1), np.ravel(x2))])
            Z = self.project(x).reshape(x1.shape)
            plt.contour(x1, x2, Z, [0.0], colors='k', linewidths=1, origin='lower')
            plt.contour(x1, x2, Z + 1, [0.0], colors='y', linewidths=1, origin='lower')
            plt.contour(x1, x2, Z - 1, [0.0], colors='y', linewidths=1, origin='lower')
            # pl.axis("tight")
            plt.title('C = {}, lambda = {}'.format(self.C, self.lmbd))
            plt.show()

    def calc_acc(self, predict, target):
        # подсчет точности
        count = np.sum(predict == target)
        return count / len(target)

    def __extend__(self, X):
        # создаем столбец из единиц нужного размера
        E = np.ones(shape=(len(X), 1))
        # Дополняем матрицу X столбцом из единиц слева
        return np.concatenate((E, X), axis=1)

    def project(self, X):
        if self.alpha is not None:
            X_ = self.__extend__(X)
            A = self.create_A(self.X_extended, X_)
            S = (A.T * (self.Y * self.alpha).T).T
            proj = np.sum(S, axis=0)
            return proj.reshape((len(proj), 1))

    def __project__(self, X, X_extended, Y, alpha):
        X_ = self.__extend__(X)
        A = self.create_A(X_extended, X_)
        S = (A.T * (Y * alpha).T).T
        proj = np.sum(S, axis=0)
        return proj.reshape((len(proj), 1))

    def predict(self, X):
        return np.sign(self.project(X))


if __name__ == '__main__':
    # размерность
    n = 2
    # количество примеров
    l = 500

    # генерация нормально-распределенных двумерных точек со средним значением = 0.5, станд.отклонением = 0.08
    # к сгенерированным точкам прибавляем смещение относительно осей X1 и X2
    # получаем объекты в таком виде: [X1, X2, Y], где X1, X2 - входные признаки объекта, Y - класс объекта

    # первое облако точек с метками 1
    class_one_1 = np.array(
        [np.concatenate((np.random.normal(loc=0.5, scale=0.08, size=n) + np.array([0.15, 0.075]), [1])) for i in
         range(int(l / 4))])

    # второе облако точек с метками 1
    class_one_2 = np.array(
        [np.concatenate((np.random.normal(loc=0.5, scale=0.08, size=n) + np.array([-0.175, 0.175]), [1])) for i in
         range(int(l / 4))])

    # сливаем в один набор точек с метками 1
    class_one = np.concatenate((class_one_1, class_one_2), axis=0)

    # облако точек с метками -1
    class_two = np.array(
        [np.concatenate((np.random.normal(loc=0.5, scale=0.08, size=n) + np.array([-0.15, -0.1]), [-1])) for j in
         range(int(l / 2))])


    # сливаем в один датасет
    data_set = np.concatenate((class_one, class_two), axis=0)

    # перемешиваем и разбиваем на обучающую и тестовую выборки
    np.random.shuffle(data_set)
    train_n = int(len(data_set) * 0.7)
    train_data = data_set[:train_n]
    test_data = data_set[train_n:]

    # отрисовываем точки
    plt.plot(class_one[:, 0], class_one[:, 1], 'bo', label="1")
    plt.plot(class_two[:, 0], class_two[:, 1], 'ro', label="-1")

    fig = plt.gcf()
    fig.set_size_inches(7, 7)
    ax = fig.gca()

    plt.ylim((-.05, 1.15))
    plt.xlim((-.05, 1.05))
    plt.grid(True)
    legend = plt.legend(loc='upper center', shadow=True, fontsize='x-large')
    legend.get_frame().set_facecolor('#00FFCC')
    plt.show()

    print('Первые 5 точек из обучающей выборки:')
    print(data_set[:5])

    max_its = 150
    eps = 1e-4
    # informing = False
    informing = True
    left_x, right_x = -0.05, 1.05
    # plotting = (max_its+1, left_x, right_x)
    plotting = None

    # linear_kernel, polynomial_kernel, sigmoid_kernel, \
    # tanh_kernel, rbf_kernel, gauss_kernel, laplace_kernel

    kernel = laplace_kernel

    print(kernel.__doc__)
    # СОЗДАЕМ МОДЕЛЬ
    model = MaxLossSVM(C=1, lmbd=0.2, kernel=kernel)

    # ТРЕНИРУЕМ
    cost_list, accuracy_list, valid_cost_list, valid_accuracy_list = model.fit(train_data=train_data,
                                                                               test_data=test_data,
                                                                               eps=eps,
                                                                               max_iters=max_its,
                                                                               info=informing,
                                                                               plotting=plotting)
    model.plot(class_one, class_two, left_x, right_x)
    # print('alpha', model.alpha)
    # print(model.project(train_data[:, :-1]))
    print('it: {}, cost: {}, acc: {}, valid_cost:{}, valid_acc: {}'.format(len(cost_list), cost_list[-1],
                                                                           accuracy_list[-1],
                                                                           valid_cost_list[-1],
                                                                           valid_accuracy_list[-1]))
    # Рисуем график изменения целевой функции
    iters = np.arange(len(cost_list))
    plt.plot(iters, cost_list)
    plt.show()
    # print(model.alpha)
    print(np.sum(model.alpha))
    print()

    print('Исследуем решения при лямбда (lmbd) из множества {0.1, 0.2, 0.3, ... 0.8, 0.9}:')
    lmbds = [0 + 0.1*i for i in range(10)]
    for lmbd in lmbds:
        print("Lambda: ", lmbd)
        informing = False
        # plotting = (5, 0.1, 1.1)
        plotting = None
        # kernel = polynomial_kernel
        model = MaxLossSVM(C=1, lmbd=lmbd, kernel=kernel)

        cost_list, accuracy_list, valid_cost_list, valid_accuracy_list = model.fit(train_data=train_data,
                                                                                          test_data=test_data,
                                                                                          eps=eps,
                                                                                          max_iters=max_its,
                                                                                          info=informing,
                                                                                          plotting=plotting)
        model.plot(class_one, class_two, left_x, right_x)

        print('it: {}, cost: {}, acc: {}, valid_cost:{}, valid_acc: {}'.format(len(cost_list), cost_list[-1],
                                                                               accuracy_list[-1],
                                                                               valid_cost_list[-1],
                                                                               valid_accuracy_list[-1]))

        # if len(iters) > 1:
        #     plt.clf()
        #     plt.plot(iters, cost_list, label='Целевая функция на обучающей выборке')
        #     plt.plot(iters, valid_cost_list, label='Целевая функция на тестовой выборке')
        #     legend = plt.legend(loc='upper right', shadow=False, fontsize='small')
        #     legend.get_frame().set_facecolor('#00FFCC')
        #     plt.show()
        #
        #     plt.clf()
        #     plt.plot(iters, accuracy_list, label='Точность на обучающей выборке')
        #     plt.plot(iters, valid_accuracy_list, label='Точность на тестовой выборке')
        #     legend = plt.legend(loc='lower right', shadow=False, fontsize='small')
        #     legend.get_frame().set_facecolor('#00FFCC')
        #     plt.show()

        predict = model.predict(test_data[:, :-1])
        accuracy = model.calc_acc(predict, test_data[:, -1:])

        print("Точность: ", accuracy)
        print()
