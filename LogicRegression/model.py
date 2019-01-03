# -*- coding:utf-8 -*-
__author__ = "andrew"
"""
逻辑回归, 采用softmax交叉熵, 能用于多分类
"""
import numpy as np


# softmax激活函数
class SoftMax:
    @staticmethod
    def forward_propagation(z):
        """
        前向传播
        :param z:
        :return:
        """
        return np.exp(z)/np.sum(np.exp(z))


def one_hot(class_num ,label):
    """
    对实数进行one_hot操作
    :param class_num: 类别数目
    :param label: 标签
    :return:
    """
    z = np.zeros((class_num, 1))
    z[label] = 1
    return z


def shuffle(X, labels):
    """
    打乱训练样本顺序
    :param X:
    :param labels:
    :return:
    """
    z = [(x, label) for x, label in zip(X, labels)]
    np.random.shuffle(z)
    X = []
    labels = []
    for x, label in z:
        X.append(x)
        labels.append(label)
    return X, labels


class Model:
    def __init__(self, input_width, class_num, learning_rate=0.001, lamda=0):
        """
        逻辑回归模型初始化
        :param input_width: 输入数据尺寸
        :param class_num: 类别数量
        :param learning_rate: 学习率
        :param lamda: 正则化项
        """
        self.input_width = input_width
        self.class_num = class_num
        self.learning_rate = learning_rate
        self.lamda = lamda
        # 随机初始化权重
        self.W = np.random.uniform(-1e-4, 1e-4, (class_num, input_width))
        self.b = np.zeros((class_num, 1))
        self.history = []  # 存储模型误差

    def calculate_gradient(self, input_array, label):
        """
        计算一个样本情况下的梯度
        :param input_array: 输入数组
        :param label: 对应样本标签
        :return:
        """
        z = self.W@input_array + self.b
        a = SoftMax.forward_propagation(z)
        e = a - one_hot(self.class_num,  label)
        # 计算梯度
        self.grad_W = e@input_array.T + self.lamda*self.W
        self.grad_b = e

    def update(self):
        """
        更新权重
        :return:
        """
        self.W -= self.learning_rate*self.grad_W
        self.b -= self.learning_rate*self.grad_b

    def train_one_sample(self, x, label):
        """
        训练一个样本
        :param x:
        :param label:
        :return:
        """
        self.calculate_gradient(x, label)
        self.update()

    def train_multiple_sample(self, X, labels):
        for x, label in zip(X, labels):
            self.train_one_sample(x, label)

    def train(self, X, labels, batch_size, epochs):
        """
        模型训练, 批量梯度下降
        :param X:
        :param labels:
        :param batch_size: 批大小
        :param epochs:
        :return:
        """
        m = len(labels)
        for epoch in range(epochs):
            # 每迭代10次, 打印一次训练情况
            if epoch%10 == 0:
                loss = self.calculate_total_loss(X, labels)
                accuracy = self.evaluate(X, labels)
                print("epoch_{}--------------------loss: {}, accuracy: {:.2%}".format(epoch, loss, accuracy))
                self.history.append((epoch, loss))
            X, labels = shuffle(X, labels) # 打乱样本顺序
            a = 0
            while a < m:
                if a+batch_size <= m:
                    self.train_multiple_sample(X[a:a+batch_size], labels[a:a+batch_size])
                else:
                    self.train_multiple_sample(X[a:-1], labels[a:-1])
                a += batch_size

    def predict(self, predict_X):
        """
        预测分类
        :param predict_X: 预测样本
        :return:
        """
        labels = []
        for x in predict_X:
            z = self.W@x + self.b
            a = SoftMax.forward_propagation(z)
            label = np.argmax(a)
            labels.append(label)
        return labels

    def evaluate(self, X, labels):
        """
        评估模型精度
        :param X:
        :param labels:
        :return:
        """
        predict_labels = self.predict(X)
        m = len(labels)
        correct = 0
        for i in range(m):
            if labels[i] == predict_labels[i]:
                correct += 1
        return correct/m

    def calculate_loss(self, x, label):
        """
        计算交叉熵损失, 一个样本
        :param x:
        :param label:
        :return:
        """
        z = self.W@x + self.b
        a = SoftMax.forward_propagation(z)
        loss = 0 - np.log(a[label][0])
        return loss

    def calculate_total_loss(self, X, labels):
        """
        计算总的交叉熵损失
        :param X:
        :param labels:
        :return:
        """
        loss = 0
        m = len(labels)
        for x, label in zip(X, labels):
            loss += self.calculate_loss(x, label)
        return loss/m

    def gradient_check(self, x, label):
        """
        梯度检查
        :return:
        """
        m = self.class_num
        n = x.shape[0]
        self.calculate_gradient(x, label)
        grad_W = self.grad_W  # 模型计算出的误差
        W = self.W
        epolision = 1e-5
        for i in range(m):
            for j in range(n):
                self.W[i][j] += epolision
                loss1 = self.calculate_loss(x, label)
                self.W[i][j] -= 2*epolision
                loss2 = self.calculate_loss(x, label)
                self.W[i][j] += epolision  # 还原self.W
                expected_gradient = (loss1 - loss2)/(2*epolision)
                diff = expected_gradient - grad_W[i][j]
                if diff != 0:
                    print("W[{}][{}]: expected gradient{} - modoel gradient{} = {}".format(i, j, expected_gradient, grad_W[i][j], diff))  # 打印梯度检查结果


if __name__ == "__main__":
    import tensorflow as tf

    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    x_train = x_train.reshape(x_train.shape[0], x_train.shape[1] * x_train.shape[2], 1)
    x_test = x_test.reshape(x_test.shape[0], x_test.shape[1] * x_test.shape[2], 1)

    # 实例化模型
    model = Model(784, 10, learning_rate=0.01, lamda=0)
    # model.gradient_check(x_train[0], y_train[0])
    model.train(x_train, y_train, 56, 100)