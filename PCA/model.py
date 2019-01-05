# -*- coding:utf-8 -*-
__author__ = "andrew"
"""
PCA算法, 降维算法
"""
import numpy as np
import matplotlib.pyplot as plt


class Model:
    def __init__(self, X, K):
        """
        pca模型初始化
        :param X: 训练样本
        :param K: 保留主成分数目
        """
        avg = np.mean(X)
        std = np.std(X)
        # 对原始数据去中心化
        self.X = (X - avg)/std
        self.K = K

    def pca(self):
        """
        pca算法实现
        :return:
        """
        XCov = np.cov(self.X, rowvar=False)  # 求字段间的协方差
        self.w, self.V = np.linalg.eig(XCov)  # 求解协方差矩阵的特征值与特征向量
        indicies = np.argsort(self.w)[::-1]  # 得到特征值从大到小排序的索引
        self.V = self.V[:, indicies]  # 将特征向量按照特征值大小从大到小排列
        Z = self.X@self.V[:, :self.K]  # 对数据降维
        return Z

    def recover(self, Z):
        """
        pca数据还原
        :return:
        """
        XRec = Z@self.V[:, :self.K].T
        return XRec


def displayData(X, row, col):
    fig, axs = plt.subplots(row, col, figsize=(8, 8))
    for r in range(row):
        for c in range(col):
            axs[r][c].imshow(X[r * col + c].reshape(32, 32).T, cmap='Greys_r')
            axs[r][c].set_xticks([])
            axs[r][c].set_yticks([])
    plt.show()

if __name__ == "__main__":
    from scipy import io
    X = io.loadmat("ex7faces.mat")['X']
    model = Model(X, 100)
    Z = model.pca()
    displayData(model.V[:, :36].T, 6, 6)
    XRec = model.recover(Z)
    displayData(XRec, 10, 10)