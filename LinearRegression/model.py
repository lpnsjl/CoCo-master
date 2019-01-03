# -*- coding:utf-8 -*-
__author__ = "andrew"
# 线性回归模型
import numpy as np
from sklearn import datasets


class Model:
    def __init__(self, X, y, learning_rate=0.001, lamda=0):
        """
        线性回归模型初始化
        :param X: 训练样本
        :param y: 训练标签
        :param learning_rate: 学习率
        :param lamda: 正则化项
        """
        self.learning_rate = learning_rate
        self.lamda = lamda
        m, n = X.shape  # m 样本个数, n 特征维数
        self.y = y.reshape(-1, 1)
        ones = np.ones((m, 1))
        self.X = np.concatenate((ones, X), axis=1)
        # 初始化权重参数
        self.theta = np.zeros((n+1, 1))
        self.history = []

    def calculate_gradient(self):
        """
        计算梯度
        :return:
        """
        m = self.y.shape[0] # 样本数量
        e = self.X@self.theta - self.y  # 计算误差
        self.grad_theta = self.X.T@e/m + self.lamda*self.theta/m  # theta的导数, 加入了正则化项

    def update(self):
        """
        更新权重
        :return:
        """
        self.theta -= self.learning_rate*self.grad_theta

    def train(self, epochs):
        """
        训练线性回归模型
        :param epochs: 迭代次数
        :return:
        """
        for epoch in range(epochs):
            # 每迭代十次, 打印一次误差
            if epoch%10 == 0:
                loss = self.calculate_loss()
                self.history.append((epoch, loss))
                print("epoch_{}--------------------- loss: {}".format(epoch, loss))
            self.calculate_gradient()
            self.update()

    def calculate_loss(self):
        """
        计算误差
        :return:
        """
        m = self.y.shape[0]  # 样本数量
        e = self.X@self.theta - self.y
        loss = 0.5*e.T@e/m + 0.5*self.lamda*self.theta.T@self.theta
        return loss[0][0]

    def predict(self, predict_X):
        """
        使用线性回归模型进行预测
        :param predict_X: 预测样本
        :return: 预测值
        """
        m, n = predict_X.shape
        ones = np.ones((m, 1))
        predict_X = np.concatenate((ones, predict_X), axis=1)
        predict_y = predict_X@self.theta
        return predict_y


if __name__ == "__main__":
    diabetes = datasets.load_diabetes()
    train_X = diabetes.get("data")
    train_y = diabetes.get("target")

    model = Model(train_X, train_y, learning_rate=1.95, lamda=0)
    model.train(100000)
