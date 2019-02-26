# -*- coding:utf-8 -*-
__author__ = 'andrew'
"""
扩展卡尔曼滤波ekf
"""


import numpy as np
import matplotlib.pyplot as plt


def filter():
    """
    卡尔曼滤波
    :return:
    """
    kx = 0.01; ky = 0.05 # 阻尼系数
    g = 9.8 # 重力加速度
    t = 15 # 仿真时间
    Ts = 0.1 # 采样周期
    step = int(t/Ts) # 仿真步数
    dax = 3; day = 3 # 系统噪声
    X = np.zeros((step, 4)) # 真实值
    X[0] = [0, 50, 500, 0] # 状态模拟的初值
    for k in range(1, step):
        x = X[k-1, 0]; vx = X[k-1, 1]; y =  X[k-1, 2]; vy = X[k-1, 3]
        x = x + vx*Ts
        vx = vx + (-kx*vx**2+dax*np.random.randn(1))*Ts
        y = y + vy*Ts
        vy = vy + (ky*vy**2-g+day*np.random.randn(1))*Ts
        X[k] = [x, vx, y, vy]
    dr = 8; dafa=0.1 # 量测噪声
    Z = np.zeros((step, 2)) # 测量值
    for k in range(step):
        r = np.sqrt(X[k, 0]**2+X[k, 2]**2) + dr*np.random.randn(1)
        a = np.tanh(X[k, 0]/X[k, 2])*57.3 + dafa*np.random.randn(1)
        Z[k] = [r, a]

    Q = np.diag([0, dax/10, 0, day/10])**2
    R = np.diag([dr, dafa])**2
    P = 10*np.eye(4)
    Pminus = 10*np.eye(4)
    x_hat = [0, 40, 400, 0]
    x_est = np.zeros((step, 4)) # 估计值
    x_forecast = np.zeros((4, 1)) # 预测值
    z = np.zeros((4, 1))
    I = np.eye(4) # 单位矩阵
    for k in range(step):
        # 状态预测
        x1 = x_hat[0] + x_hat[1]*Ts
        vx1 = x_hat[1] + (-kx*x_hat[1]**2)*Ts
        y1 = x_hat[2] + x_hat[3]*Ts
        vy1 = x_hat[3] + (ky*x_hat[3]**2 - g)*Ts
        x_forecast[:, 0] = [x1, vx1, y1, vy1]

        # 测量值预测
        r = np.sqrt(x1*x1 + y1*y1)
        alpha = np.tanh(x1/y1)*57.3
        y_yuce = np.array([r, alpha]).reshape(-1, 1)

        # 状态矩阵
        vx = x_forecast[1]; vy = x_forecast[3]
        F = np.zeros((4, 4))
        F[0, 0] = 1; F[0, 1] = Ts
        F[1, 1] = 1-2*kx*vx*Ts
        F[2, 2] = 1; F[2, 3] = Ts
        F[3, 3] = 1+2*ky*vy*Ts
        Pminus = F@P@F.T + Q
        # 测量值矩阵
        x = x_forecast[0]; y = x_forecast[2]
        H = np.zeros((2, 4))
        r = np.sqrt(x**2+y**2); xy2 = 1+(x/y)**2
        H[0, 0] = x/r; H[0, 2] = y/r
        H[1, 0] = (1/y)/xy2; H[1, 2] = (-x/y**2)/xy2
        K = Pminus@H.T@np.linalg.inv(H@Pminus@H.T+R)
        x_hat = x_forecast + K@(Z[k].reshape(-1, 1) - y_yuce)
        # print(x_hat.flatten())
        P = (I - K@H)@Pminus
        x_est[k] = x_hat.flatten()

    plt.plot(X[:, 0], X[:, 2], 'g')
    plt.plot(Z[:, 0]*np.sin(Z[:, 1]*np.pi/180), Z[:, 0]*np.cos(Z[:, 1]*np.pi/180))
    print(x_est)
    plt.plot(x_est[:, 0], x_est[:, 2], 'r')
    plt.show()

if __name__ == '__main__':
    filter()


