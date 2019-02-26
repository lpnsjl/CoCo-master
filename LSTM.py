# -*- coding:utf-8 -*-
__author__ = "andrew"
"""
标准LSTM网络(只有一层隐藏层)
"""
import numpy as np
from activation import *
from prepend.load_starndardrnn_data import loadStandardRNNData


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


class LSTM(object):
    """
    LSTM网络
    """
    def __init__(self, input_width, hidden_size=100):
        """
        LSTM网络初始化
        :param input_width: 输入宽度
        :param hidden_size: 隐藏层神经元个数
        """
        self.input_width = input_width
        self.hidden_size = hidden_size

        self.V = np.random.uniform(-1e-4, 1e-4, (input_width, hidden_size))
        self.c = np.zeros((input_width, 1))
        # 遗忘门参数
        self.Wf = np.random.uniform(-1e-4, 1e-4, (hidden_size, hidden_size))
        self.Uf = np.random.uniform(-1e-4, 1e-4, (hidden_size, input_width))
        self.bf = np.zeros((hidden_size, 1))
        # 输入门参数
        self.Wi = np.random.uniform(-1e-4, 1e-4, (hidden_size, hidden_size))
        self.Ui = np.random.uniform(-1e-4, 1e-4, (hidden_size, input_width))
        self.bi = np.zeros((hidden_size, 1))

        self.Wa = np.random.uniform(-1e-4, 1e-4, (hidden_size, hidden_size))
        self.Ua = np.random.uniform(-1e-4, 1e-4, (hidden_size, input_width))
        self.ba = np.zeros((hidden_size, 1))
        # 输出门参数
        self.Wop = np.random.uniform(-1e-4, 1e-4, (hidden_size, hidden_size))
        self.Uop = np.random.uniform(-1e-4, 1e-4, (hidden_size, input_width))
        self.bop = np.zeros((hidden_size, 1))

    def forward_propagation(self, sentence):
        """
        前向传播
        :param sentence: 输入句子
        :return:
        """
        S = [] # 保存状态值
        C = []  # 保存细胞状态
        # 初始状态与初始细胞状态, 并放入列表中
        s = np.zeros((self.hidden_size, 1))
        c = np.zeros((self.hidden_size, 1))
        S.append(s)
        C.append(c)
        O = [] # 保存输出值
        F = [] # 保存遗忘门值
        OP = [] # 保存输出门值
        # 保存输入门值
        A = []
        I = []

        T = len(sentence)  # 句子的单词数量, 实际上表示有多少个时刻
        for t in range(T):
            x = one_hot(self.input_width, sentence[t])  # one_hot该时刻输入
            # 计算遗忘门
            f = Sigmoid.forward(self.Wf@s + self.Uf@x + self.bf)
            F.append(f)
            # 计算输入门
            i = Sigmoid.forward(self.Wi@s + self.Ui@x + self.bi)
            a = TanhActivator.forward_propagation(self.Wa@s + self.Ua@x + self.ba)
            I.append(i)
            A.append(a)
            # 计算细胞状态
            c = f*c + i*a
            C.append(c)
            # 计算输出门
            op = Sigmoid.forward(self.Wop@s + self.Uop@x + self.bop)
            OP.append(op)
            # 计算状态值
            s = op*TanhActivator.forward_propagation(c)
            S.append(s)
            # 计算输出
            o = SoftmaxActivator.forward_propagation(self.V@s + self.c)
            O.append(o)
        return O, S, C, F, I, A, OP

    def backward_propagation(self, sentence, label):
        """
        反向传播
        :param sentence: 输入句子
        :param label: 真实标签
        :return:
        """
        self.grad_Wf = np.zeros(self.Wf.shape)
        self.grad_Uf = np.zeros(self.Uf.shape)
        self.grad_bf = np.zeros(self.bf.shape)

        self.grad_Wi = np.zeros(self.Wi.shape)
        self.grad_Ui = np.zeros(self.Ui.shape)
        self.grad_bi = np.zeros(self.bi.shape)

        self.grad_Wa = np.zeros(self.Wa.shape)
        self.grad_Ua = np.zeros(self.Ua.shape)
        self.grad_ba = np.zeros(self.ba.shape)

        self.grad_Wop = np.zeros(self.Wop.shape)
        self.grad_Uop = np.zeros(self.Uop.shape)
        self.grad_bop = np.zeros(self.bop.shape)

        self.grad_V = np.zeros(self.V.shape)
        self.grad_c = np.zeros(self.c.shape)

        T = len(sentence)
        O, S, C, F, I, A, OP = self.forward_propagation(sentence)
        e = O[-1] - one_hot(self.input_width, label[-1]) # 最后时刻的误差
        delta_ht = self.V.T@e
        delta_ct = delta_ht*OP[-1]*(1 - np.tanh(C[-1])**2)
        for t in range(T)[::-1]:
            x = one_hot(self.input_width, sentence[t])  # one_hot该时刻输入
            self.grad_V += e@S[t+1].T
            self.grad_c += e
            # 遗忘门
            f_midval = delta_ct*C[t]*F[t]*(1-F[t]) # 计算遗忘门参数的中间值
            self.grad_Wf += f_midval@S[t].T
            self.grad_Uf += f_midval@x.T
            self.grad_bf += f_midval
            # 输入门
            i_midval = delta_ct*A[t]*I[t]*(1-I[t])
            self.grad_Wi += i_midval@S[t].T
            self.grad_Ui += i_midval@x.T
            self.bi += i_midval

            a_midval = delta_ct*I[t]*(1 - np.tanh(A[t])**2)
            self.grad_Wa += a_midval@S[t].T
            self.grad_Ua += a_midval@x.T
            self.grad_ba += a_midval
            # 输出门
            op_midval = delta_ht*np.tanh(C[t+1])*OP[t]
            self.grad_Wop += op_midval@S[t].T
            self.grad_Uop += op_midval@x.T
            self.grad_bop += op_midval

            if t != 0:
                e = O[t-1] - one_hot(self.input_width, label[t-1])
                delta_ht = self.V.T@e
                delta_ct = delta_ht*OP[t-1]*(1 - np.tanh(C[t])**2) + delta_ct*F[t]

    def update(self, learn_rate):
        """
        更新参数
        :param learn_rate: 学习率
        :return:
        """
        self.V -= learn_rate * self.grad_V
        self.c -= learn_rate * self.grad_c

        self.Wf -= learn_rate * self.grad_Wf
        self.Uf -= learn_rate * self.grad_Uf
        self.bf -= learn_rate * self.grad_bf

        self.Wi -= learn_rate * self.grad_Wi
        self.Ui -= learn_rate * self.grad_Ui
        self.bi -= learn_rate * self.grad_bi

        self.Wa -= learn_rate * self.grad_Wa
        self.Ua -= learn_rate * self.grad_Ua
        self.ba -= learn_rate * self.grad_ba

        self.Wop -= learn_rate * self.grad_Wop
        self.Uop -= learn_rate * self.grad_Uop
        self.bop -= learn_rate * self.grad_bop

    def train_one_sample(self, sentence, label, learn_rate):
        """
        训练一个样本
        :param sentence:
        :param label:
        :param learn_rate:
        :return:
        """
        self.backward_propagation(sentence, label)
        self.update(learn_rate)

    def train(self, X_train, y_train, epochs, learn_rate):
        """
        模型训练
        :param X_train: 训练样本
        :param y_train: 训练标签
        :param learn_rate: 学习率
        :param epochs: 迭代次数
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
        O, S, C, F, I, A, OP = self.forward_propagation(sentence)
        for o, l in zip(O, label):
            loss += -1*(one_hot(self.input_width, l).T@np.log(o))[0][0]
        return loss

    def predict(self, sentence):
        """
        模型预测
        :param sentence: 输入句子
        :return: 模型预测出的新句子
        """
        O, S, C, F, I, A, OP = self.forward_propagation(sentence)
        O = np.array(O)
        return np.argmax(O, axis=1)


if __name__ == "__main__":
    vocabulary_size = 8000
    X_train, y_train = loadStandardRNNData("./data/reddit-comments-2015-08.csv", vocabulary_size)
    model = LSTM(vocabulary_size, 100)  # 创建模型实例
    model.train(X_train[:100], y_train[:100], 100, learn_rate=0.005)  # 训练模型
