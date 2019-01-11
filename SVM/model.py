"""
svm支持向量机
"""
import numpy as np
from itertools import cycle
import matplotlib.pyplot as plt


class Sigmoid:
    @staticmethod
    def forward_propagation(z):
        return 1/(1+np.exp(-z))


def calcK(X, z, KernelPara):
    """
    计算一个样本的核函数值
    :param X:
    :param z:
    :param KernelPara: 核函数参数
    :return:
    """
    m, n = X.shape
    K = np.zeros((m, 1))
    # 线性核
    if KernelPara[0] == "linear":
        K = X@z
        return K
    # 径向核, KernelPara第二个参数是径向核参数
    elif KernelPara[1] == "rbf":
        for j in range(m):
            x = X[j].reshape(-1, 1)
            xx = (x - z).T@(x - z)
            K[j, 0] = xx[0][0]
        K = np.exp(-0.5*K/KernelPara[1]**2)
        return K
    else:
        raise Exception("use correct kernel: 'linear'/'rbf'")


class SVM:
    def __init__(self, X_train, y_train, C, tol, KernelPara):
        """
        svm初始化
        :param X_train: 训练样本
        :param y_train: 训练标签
        :param C: 软间隔参数
        :param tol: kkt条件容忍度
        :param KernelPara: 核函数参数
        """
        self.m, self.n = X_train.shape
        self.X_train = X_train
        # 改变标签类别
        y = y_train.copy()
        y[y==0] = -1
        print(y)
        self.y_train = y.reshape(-1, 1)
        print("y_train: {}".format(self.y_train))
        self.C = C
        self.tol = tol
        self.KernelPara = KernelPara
        self.alphas = np.zeros((self.m, 1))
        self.b = 0
        self.eCache = np.zeros((self.m, 2))
        self.K = np.zeros((self.m, self.m))  # 存储核函数结果
        for j in range(self.m):
            self.K[:, j] = calcK(X_train, X_train[j].reshape(-1, 1), KernelPara).flatten()

    def calcEk(self, k):
        """
        计算Ek
        :param k:
        :return:
        """
        gk = (self.alphas*self.y_train).T@self.K[:, k].reshape(-1, 1) + self.b
        ek = gk - self.y_train[k]
        return ek[0][0]

    def updateEk(self, k, ek):
        """
        更新eCache
        :param k:
        :param ek:
        :return:
        """
        self.eCache[k] = [1, ek]

    def selectJRand(self, i):
        """
        随机选取j
        :param i:
        :return:
        """
        j = np.random.randint(0, self.m)
        while j == i:
            j = np.random.randint(0, self.m)
        return j

    def selectJ(self, i):
        """
        选取j, 使得|ei - ej|值最大, 主要是为了加快smo算法迭代速度
        :param i:
        :return:
        """
        ei = self.calcEk(i)  # 计算ei
        self.updateEk(i, ei)
        validEcacheList = np.nonzero(self.eCache[:, 0])[0]
        maxDeltaEk = 0
        maxJ = self.selectJRand(i)
        for j in validEcacheList:
            if j != i:
                ej = self.calcEk(j)
                deltaEk = abs(ej - ei)
                if deltaEk > maxDeltaEk:
                    maxJ = j
        j = maxJ
        ej = self.calcEk(j)
        return j, ej

    @staticmethod
    def clipAlpha(aj, L, H):
        """
        修剪alpha
        :param aj:
        :param L:
        :param H:
        :return:
        """
        if aj < L:
            aj = L
        if aj >H:
            aj = H
        return aj

    def innerL(self, i):
        """
        smo算法内循环
        :param i:
        :return:
        """
        ei = self.calcEk(i)
        # 假如ai违反了kkt条件, 对其进行二次规划
        if ((self.y_train[i]*ei < -self.tol) and (self.alphas[i] < self.C)) or ((self.y_train[i]*ei > self.tol) and (self.alphas[i] > 0)):
            j, ej = self.selectJ(i)
            aiOld = self.alphas[i, 0]
            ajOld = self.alphas[j, 0]
            if self.y_train[i] != self.y_train[j]:
                L = max(0, ajOld-aiOld)
                H = min(self.C, self.C+ajOld-aiOld)
            else:
                L = max(0, ajOld+aiOld-self.C)
                H = min(self.C, ajOld+aiOld)
            if L==H:
                return 0
            eta = self.K[i, i] + self.K[j, j] - 2*self.K[i, j]
            if eta <= 0:
                return 0
            # 更新aj, ai; 如果更新幅度小于阀值, 则认为已经不能再更新了
            self.alphas[j] += self.y_train[j]*(ei-ej)/eta
            self.alphas[j] = self.clipAlpha(self.alphas[j], L, H)
            self.updateEk(j, ej)
            if abs(self.alphas[j] - ajOld) < 0.001:
                return 0
            self.alphas[i] += self.y_train[i]*self.y_train[j]*(ajOld - self.alphas[j])
            # 计算b
            b1 = self.b - ei - self.y_train[i]*self.K[i, i]*(self.alphas[i] - aiOld) - self.y_train[j]*self.K[j, i]*(self.alphas[j] - ajOld)
            b2 = self.b - ej - self.y_train[i]*self.K[i, j]*(self.alphas[i] - aiOld) - self.y_train[j]*self.K[j, j]*(self.alphas[j] - ajOld)
            if 0<self.alphas[i]<self.C:
                self.b = b1
            elif 0<self.alphas[j]<self.C:
                self.b = b2
            else:
                self.b = (b1+b2)/2
            return 1
        else:
            return 0

    def smo(self, epochs):
        """
        smo算法
        :return:
        """
        epoch = 0
        entireSet = True
        alphaPairsChanged = 0
        while (epoch < epochs) and ((alphaPairsChanged > 0) or entireSet):
            alphaPairsChanged = 0
            if entireSet:
                for i in range(self.m):  # 遍历所有数据
                    alphaPairsChanged += self.innerL(i)
                    print("fullSet, epoch: %d i:%d, pairs changed %d" % (
                    epoch, i, alphaPairsChanged))  # 显示第多少次迭代，那行特征数据使alpha发生了改变，这次改变了多少次alpha
                epoch += 1
            else:
                nonBoundIs = np.nonzero((self.alphas > 0) * (self.alphas < self.C))[0]
                for i in nonBoundIs:  # 遍历非边界的数据
                    alphaPairsChanged += self.innerL(i)
                    print("non-bound, epoch: %d i:%d, pairs changed %d" % (epoch, i, alphaPairsChanged))
                epoch += 1
            if entireSet:
                entireSet = False
            elif alphaPairsChanged == 0:
                entireSet = True
            print("epochation number: %d" % epoch)

    def predict(self, X):
        """
        预测函数
        :param X: 预测样本
        :return:
        """
        m, n = X.shape
        predict_ys = np.zeros(m)
        for i in range(m):
            Ki = calcK(self.X_train, X[i].reshape(-1, 1), self.KernelPara)
            z = (self.alphas*self.y_train).T@Ki + self.b
            predict_y = Sigmoid.forward_propagation(z[0][0])
            if predict_y >= 0.5:
                predict_ys[i] = 1
            else:
                predict_ys[i] = 0
        return predict_ys

    def evaluate(self, X, labels):
        """
        评估模型准确率
        :param X: 测试样本
        :param labels: 测试标签
        :return:
        """
        m = labels.shape[0]
        predict_ys = self.predict(X)
        correctNum = 0
        for i in range(m):
            if labels[i] == predict_ys[i]:
                correctNum += 1
        accuracy = correctNum/m
        print("accuracy: {:.2%}".format(accuracy))


def displayData(svm, X, y):
    # ys = svm.predict(X)
    ys = y
    index = (ys==0)
    member = X[index]
    plt.plot(member[:, 0], member[:, 1], 'r+')
    index = (ys==1)
    member = X[index]
    plt.plot(member[:, 0], member[:, 1], 'b.')
    maxValue = X[:, 0].max()
    minValue = X[:, 0].min()
    w = svm.X_train.T@(svm.y_train*svm.alphas)
    x = np.linspace(minValue, maxValue, 50)
    y = -1*(w[0, 0]*x+svm.b)/w[1, 0]
    plt.plot(x, y, 'g')
    plt.show()


if __name__ == "__main__":
    from scipy import io
    data = io.loadmat("ex6data1.mat")
    X = data['X']
    y = data['y'].flatten()
    y = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1])
    svm = SVM(X, y, 20, 0.001, ['linear'])
    svm.smo(1000)
    svm.evaluate(X, y)
    displayData(svm, X, y)