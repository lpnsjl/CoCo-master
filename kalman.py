# -*- coding:utf-8 -*-
__author__ = 'andrew'
"""
卡尔曼滤波算法, 匀加速运动的例子
"""
import numpy as np
import matplotlib.pyplot as plt


def filter():
    delta_t = 0.1
    t = np.arange(0, 5, delta_t).reshape(-1, 1)
    m = t.size
    g = 10 # 加速度
    x = 1/2*g*np.square(t) # 真实值, 实际上是我们的预测值
    z = x + np.sqrt(10)*np.random.randn(m, 1)

    A = np.array([[1, delta_t], [0, 1]]) # 状态矩阵
    B = np.array([[1/2*np.square(delta_t)], [delta_t]]) # 转换矩阵
    H = np.array([[1, 0]])

    Q = np.array([[0, 0], [0, 9e-1]]) # 状态误差
    R = np.array([[10]]) # 量测误差

    xhat = np.zeros((2, m)) # 后验估计
    P = np.zeros((2, 2)) # 真实值与估计值之间的协方差
    xhatminus = np.zeros((2, m)) # 前验估计
    Pminus = np.zeros((2, 2)) # 真实值与预测值之间的协方差
    I = np.eye(2)

    for k in range(1, m):
        # 预测
        xhatminus[:, k:k+1] = A@xhat[:, k-1:k] + B*g
        Pminus = A@P@A.T + Q

        # 更新
        K = Pminus@H.T@np.linalg.inv(H@Pminus@H.T + R)
        print(z[k:k+1].shape)
        xhat[:, k:k+1] = xhatminus[:, k:k+1] + K@(z[k:k+1] - H@xhatminus[:, k:k+1])
        P = (I - K@H)@Pminus

    plt.plot(t, z, 'b')
    plt.plot(t, xhat[0], 'r')
    plt.plot(t, x, 'g')
    plt.show()

filter()
