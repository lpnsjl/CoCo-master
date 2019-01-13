"""
svm支持向量机
"""
import numpy as np
import matplotlib.pyplot as plt


class Sigmoid:
    @staticmethod
    def forward_propagation(z):
        return 1/(1+np.exp(-z))


def calcK(X, z, KernelPara):
    """
    计算核函数
    :param X: 所有训练样本
    :param z: 某个训练样本
    :param KernelPara: 核函数
    :return:
    """
    # 线性核
    if KernelPara[0] == "linear":
        K = X@z
        return K
    # 高斯核
    elif KernelPara[0] == "rbf":
        m = X.shape[0]
        K = np.zeros((m, 1))
        for j in range(m):
            x = X[j].reshape(-1, 1)
            diff = (x - z).T@(x - z)
            K[j, 0] = np.exp(-0.5*diff[0, 0]/KernelPara[1]**2)
        return K
    else:
        raise Exception("please use correct Kernel, 'linear'/'rbf'")


class SVM:
    def __init__(self, X_train, y_train, C, tol, KernelPara):
        """
        SVM初始化
        :param X_train: 训练样本
        :param y_train: 训练标签
        :param C: 软间隔
        :param tol: kkt容忍度
        :param KernelPara: 核函数
        """
        self.m, self.n = X_train.shape
        self.X_train = X_train
        y_train = y_train.copy()
        y = np.zeros(self.m)
        for i in range(self.m):
            if y_train[i] == 0:
                y[i] = -1
            else:
                y[i] = 1
        self.y_train = y.reshape(-1, 1)
        self.C = C
        self.tol = tol
        self.b = 0
        self.alphas = np.zeros((self.m, 1))
        self.eCache = np.zeros((self.m, 2))
        self.K = np.zeros((self.m, self.m))
        self.KernelPara = KernelPara
        for j in range(self.m):
            self.K[:, j] = calcK(X_train, X_train[j].reshape(-1, 1), KernelPara).flatten()

    def calcEk(self, k):
        """
        计算Ek
        :param k:
        :return:
        """
        gk = (self.alphas*self.y_train).T@self.K[:, k].reshape(-1, 1) + self.b
        Ek = gk - self.y_train[k]
        return Ek[0, 0]

    def updateEk(self, k):
        """
        更新Ek
        :param k:
        :return:
        """
        Ek = self.calcEk(k)
        self.eCache[k] = [1, Ek]

    @staticmethod
    def clipAlpha(alpha, L, H):
        """
        约束alpha
        :param alpha:
        :param L:
        :param H:
        :return:
        """
        if alpha < L:
            alpha = L
        if alpha > H:
            alpha = H
        return alpha

    def selectJ_max(self, i):
        """
        选择使得abs(Ei - Ej)最大的j
        :param i:
        :return:
        """
        Ei = self.calcEk(i)
        self.eCache[i] = [1, Ei]
        nonBounds = np.nonzero((self.alphas > 0)*(self.alphas < self.C))[0]
        for k in nonBounds:
            Ek = self.calcEk(k)
            self.eCache[k] = [1, Ek]
        validECacheList = np.nonzero(self.eCache[:, 0])[0]
        if len(validECacheList) > 1:
            maxDelta = 0
            maxJ = 0
            for j in validECacheList:
                Ej = self.calcEk(j)
                delta = abs(Ei - Ej)
                if delta > maxDelta:
                    maxJ = j
                    maxDelta = delta
            return maxJ
        return i

    @staticmethod
    def selectJ_random(i, alls):
        """
        从所有样本中随机选取j
        :param i:
        :param alls:
        :return:
        """
        m = len(alls)
        j = np.random.randint(0, m)
        return alls[j]


    @staticmethod
    def selectJ_support(i, supports):
        """
        从支持向量中选取j
        :param i:
        :param supports:
        :return:
        """
        m = len(supports)
        j = np.random.randint(0, m)
        return supports[j]


    def take_one_step(self, i, j):
        """
        更新一次
        :param i:
        :param j:
        :return:
        """
        Ei = self.calcEk(i)
        Ej = self.calcEk(j)
        aiOld = self.alphas[i, 0]
        ajOld = self.alphas[j, 0]
        if self.y_train[i] == self.y_train[j]:
            L = max(0, aiOld+ajOld-self.C)
            H = min(self.C, aiOld+ajOld)
        else:
            L = max(0, ajOld-aiOld)
            H = min(self.C, self.C+ajOld-aiOld)
        if L == H:
            return 0
        eta = self.K[i, i] + self.K[j, j] - 2*self.K[i, j]
        if eta <= 0:
            return 0
        self.alphas[j] = ajOld + self.y_train[j]*(Ei-Ej)/eta
        self.alphas[j] = self.clipAlpha(self.alphas[j], L, H)
        if abs(self.alphas[j]-ajOld) < 0.001:
            return 0
        self.alphas[i] = aiOld + self.y_train[i]*self.y_train[j]*(ajOld-self.alphas[j])
        # 计算b
        b1 = -Ei - self.y_train[i]*self.K[i, i]*(self.alphas[i]-aiOld) - self.y_train[j]*self.K[j, i]*(self.alphas[j]-ajOld) + self.b
        b2 = -Ej - self.y_train[i]*self.K[i, j]*(self.alphas[i]-aiOld) - self.y_train[j]*self.K[j, j]*(self.alphas[j]-ajOld) + self.b
        if 0<self.alphas[i]<self.C:
            self.b = b1
        elif 0<self.alphas[j]<self.C:
            self.b = b2
        else:
            self.b = (b1+b2)/2
        self.updateEk(i)
        self.updateEk(j)
        return 1  # 更新成功

    def innerL(self, i):
        """
        内循环
        :param i:
        :return:
        """
        Ei = self.calcEk(i)
        # 违反KKT条件
        if((self.y_train[i]*Ei < -self.tol) and (self.alphas[i] < self.C)) or ((self.y_train[i]*Ei > self.tol) and (self.alphas[i] > 0)):
            j = self.selectJ_max(i)
            if j != i:
                if self.take_one_step(i, j):
                    return 1
            supports = list(np.nonzero((self.alphas>0)*(self.alphas<self.C))[0])
            while supports:
                j = self.selectJ_support(i, supports)
                if j == i:
                    supports.remove(j)
                    continue
                else:
                    if self.take_one_step(i, j):
                        return 1
                    supports.remove(j)


            alls = list(range(0, len(self.alphas)))
            while alls:
                j = self.selectJ_random(i, alls)
                if j == i:
                    alls.remove(j)
                else:
                    if self.take_one_step(i, j):
                        return 1
                    alls.remove(j)
        return 0

    def smo(self, epochs):
        """
        smo算法
        :param epochs:
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
                    # print("fullSet, epoch: %d i:%d, pairs changed %d" % (
                    #     epoch, i, alphaPairsChanged))  # 显示第多少次迭代，那行特征数据使alpha发生了改变，这次改变了多少次alpha
                epoch += 1
            else:
                nonBoundIs = np.nonzero((self.alphas > 0) * (self.alphas < self.C))[0]
                for i in nonBoundIs:  # 遍历非边界的数据
                    alphaPairsChanged += self.innerL(i)
                    #print("non-bound, epoch: %d i:%d, pairs changed %d" % (epoch, i, alphaPairsChanged))
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
            z = (self.alphas * self.y_train).T @ Ki + self.b
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
        accuracy = correctNum / m
        print("accuracy: {:.2%}".format(accuracy))

def displayData(svm, X, y):
    # ys = svm.predict(X)
    ys = y
    index = (ys == 0)
    member = X[index]
    plt.plot(member[:, 0], member[:, 1], 'r+')
    index = (ys == 1)
    member = X[index]
    plt.plot(member[:, 0], member[:, 1], 'b.')
    maxValue = X[:, 0].max()
    minValue = X[:, 0].min()
    w = svm.X_train.T @ (svm.y_train * svm.alphas)
    x = np.linspace(minValue, maxValue, 50)
    y = -1 * (w[0, 0] * x + svm.b) / w[1, 0]
    plt.plot(x, y, 'g')
    plt.show()

if __name__ == "__main__":
    from scipy import io
    data = io.loadmat("ex6data2.mat")
    X = data['X']
    y = data['y'].flatten()
    svm = SVM(X, y, 1, 0.001, ['rbf', 0.1])
    svm.smo(100)
    svm.evaluate(X, y)
    print(svm.alphas[svm.alphas != 0])
    print(svm.b)
    displayData(svm, X, y)
