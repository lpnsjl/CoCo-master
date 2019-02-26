"""
深度神经网络(DNN)的全连接层
"""
import numpy as np


class Layer:
    def __init__(self, input_size, output_size, forward_activator, backward_activator):
        """
        全连接层初始化
        :param input_size: 输入尺寸
        :param output_size: 输出尺寸
        :param forward_activator: 前向传播使用的激活函数
        :param backward_activator: 反向传播使用的激活函数
        """
        self.input_size = input_size
        self.output_size = output_size
        self.forward_activator = forward_activator()
        self.backward_activator = backward_activator()
        self.W = np.random.uniform(-1e-2, 1e-2, (output_size, input_size))  # 初始化W
        self.b = np.zeros((output_size, 1))

    def forward_propagation(self, A):
        """
        全连接层的前向传播
        :param A: 前一层的输出(本层的输入)
        :return: 本层的输出
        """
        self.A = A
        Z = self.W@A + self.b
        A = self.forward_activator.forward_propagation(Z)  # 激活函数
        return A

    def backward_propagation(self, dZ):
        """
        全连接层的反向传播
        :param dZ: 后一层的误差(用来计算本层的误差, 本层的误差需要传入到前一层)
        :return: 本层的误差
        """
        n, m = self.A.shape  # m指训练样本的个数
        self.dW = (dZ@self.A.T)/m  # 计算dW
        # print(self.dW)
        self.db = np.sum(dZ, axis=1, keepdims=True)/m  # 计算db
        dZ = self.W.T@dZ*self.backward_activator.backward_propagation(self.A)  # 计算前一层的误差
        return dZ

    def update(self, alpha):
        """
        更新全连接层的权重
        :param alpha: 学习率
        :return:
        """
        self.W -= alpha*self.dW
        self.b -= alpha*self.db


