# -*- coding:utf-8 -*-
__author__ = "andrew"
"""
实现一个标准rnn, 也就是只有一个隐藏层的rnn
"""
import numpy as np
from prepend.load_starndardrnn_data import loadStandardRNNData
from activation import *


def one_hot(vocabulary_size, i):
    """
    one_hot
    :param vocabulary_size: 词汇表大小
    :param i: 单词在词汇表的索引
    :return: 单词的one_hot表示
    """
    x = np.zeros((vocabulary_size, 1))
    x[i] = 1
    return x


class StandardRNN(object):
    def __init__(self, input_width, hidden_size=100):
        """
        rnn初始化
        :param input_width: 输入大小(自然语言处理, 一般指词汇表大小)
        :param hidden_size: 隐藏层神经元个数
        """
        self.input_width = input_width
        self.hidden_size = hidden_size
        self.W = np.random.uniform(-1e-4, 1e-4, (hidden_size, hidden_size))  # 状态权重
        self.U = np.random.uniform(-1e-4, 1e-4, (hidden_size, input_width))  # 输入层权重
        self.V = np.random.uniform(-1e-4, 1e-4, (input_width, hidden_size))  # 输出层权重
        self.c = np.zeros((input_width, 1))  # 输出层偏差
        self.b = np.zeros((hidden_size, 1))  # 输入层偏差

    def forward_propagation(self, sentence):
        """
        前向传播
        :param sentence: 输入句子
        :return:
        """
        T = len(sentence)  # 句子单词数量
        S = []  # 存储rnn各个时刻状态
        O = []  # 存储rnn各个时刻输出
        s = np.zeros((self.hidden_size, 1))  # 初始状态, s0
        S.append(s)
        for t in range(T):
            s = TanhActivator.forward_propagation(self.U@one_hot(self.input_width, sentence[t]) + self.W@s + self.b)  # 计算该时刻状态
            o = SoftmaxActivator.forward_propagation(self.V@s + self.c)  # 计算该时刻输出
            S.append(s)
            O.append(o)
        return O, S

    def backward_propagation(self, sentence, label):
        """
        反向传播, 计算梯度
        :param sentence: 输入句子
        :param label: 真实标签
        :return:
        """
        self.grad_c = np.zeros(self.c.shape)
        self.grad_V = np.zeros(self.V.shape)
        self.grad_b = np.zeros(self.b.shape)
        self.grad_W = np.zeros(self.W.shape)
        self.grad_U = np.zeros(self.U.shape)
        self.grad_b = np.zeros(self.b.shape)
        O, S = self.forward_propagation(sentence)
        T = len(sentence)
        e = O[-1] - one_hot(self.input_width, label[-1])  # 最后时刻的误差
        delta_t = self.V.T@e
        for t in range(T)[::-1]:
            self.grad_c += e
            self.grad_V += e@S[t+1].T
            self.grad_W += delta_t@S[t].T
            self.grad_U += delta_t@one_hot(self.input_width, sentence[t]).T
            self.grad_b += delta_t
            if t != 0:
                e = O[t-1] - one_hot(self.input_width, label[t-1])
                delta_t = self.V.T@e*TanhActivator.backward_propagation(S[t])
                + self.W.T@delta_t*TanhActivator.backward_propagation(S[t])

    def update(self, learn_rate):
        """
        更新参数
        :param learn_rate: 学习率
        :return:
        """
        self.W -= learn_rate*self.grad_W
        self.U -= learn_rate*self.grad_U
        self.c -= learn_rate*self.grad_c
        self.b -= learn_rate*self.grad_b
        self.V -= learn_rate*self.grad_V

    def train_one_sample(self, sentence, label, learn_rate):
        """
        训练一个样本
        :param sentence: 输入句子
        :param label:
        :param learn_rate:
        :return:
        """
        self.backward_propagation(sentence, label)
        self.update(learn_rate)

    def calculate_total_loss(self, X_train, y_train):
        """
        计算损失函数
        :param X_train: 训练样本
        :param y_train: 训练标签
        :return:
        """
        loss = 0
        size = len(X_train)
        for i in range(size):
            loss += self.calculate_loss(X_train[i], y_train[i])
        return loss/size

    def calculate_loss(self, sentence, label):
        """
        计算一个句子产生的误差
        :param sentence:
        :param label:
        :return:
        """
        loss = 0
        O, S = self.forward_propagation(sentence)
        for o, l in zip(O, label):
            loss += -1*(one_hot(self.input_width, l).T@np.log(o))[0][0]
        return loss

    def train(self, X_train, y_train, epochs, learn_rate):
        """
        rnn训练
        :param X_train: 所有的训练样本
        :param y_train: 所有的训练标签
        :param learn_rate:
        :param epochs:
        :return:
        """
        size = len(X_train)  # 训练样本数量
        for epoch in range(epochs):
            # 每迭代5次, 打印一次精度
            if epoch % 5 == 0:
                loss = self.calculate_total_loss(X_train, y_train)
                print("epoch: {}------------------------".format(epoch))
                print("loss: {}".format(loss))
            for i in range(size):
                self.train_one_sample(X_train[i], y_train[i], learn_rate)

    def predict(self, sentence):
        """
        模型预测
        :param sentence: 输入句子
        :return: 模型预测出的新句子
        """
        O, S = self.forward_propagation(sentence)
        O = np.array(O)
        return np.argmax(O, axis=1)


if __name__ == "__main__":
    vocabulary_size = 8000
    X_train, y_train = loadStandardRNNData("./data/reddit-comments-2015-08.csv", vocabulary_size)
    model = StandardRNN(vocabulary_size, 100)  # 创建模型实例
    model.train(X_train[:100], y_train[:100], 100, learn_rate=0.005)  # 训练模型
